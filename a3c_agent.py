from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import numpy
import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib import features

from network import build_fcn
import utils as U

MINIMAPFEATS = 17
SCREENFEATS = 23

class A3CAgent(object):
  """An agent specifically for solving the mini-game maps. Now converted to work for the DRDC Simulator."""
  def __init__(self, training, image_size, actionsNum, name='A3C/A3CAgent'):
    self.name = name
    self.training = training
    self.summary = []
    # Minimap size, screen size and info size
    self.image_size = image_size
    self.isize = actionsNum
    self.policyTensor = None
    self.valueTensor = None
    self.policyLoss = None
    self.valueLoss = None


  #return the policy loss and value loss variables to generate a graph over game iterations.
  def returngraphvalue(self):
    return self.policyLoss, self.valueLoss

  def setup(self, sess, summary_writer):
    self.sess = sess
    self.summary_writer = summary_writer


  def initialize(self):
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)


  def reset(self):
    # Epsilon schedule
    self.epsilon = [0.05, 0.2]


  def build_model(self, dev):
    with tf.variable_scope(self.name) and tf.device(dev):

      # Set inputs of networks
      self.minimap = tf.placeholder(tf.float32, [None, MINIMAPFEATS, self.image_size, self.image_size], name='minimap')
      self.screen = tf.placeholder(tf.float32, [None, SCREENFEATS, self.image_size, self.image_size], name='screen')
      self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')

      # Build networks
      net = build_fcn(self.minimap, self.screen, self.info, self.image_size, self.isize)
      self.spatial_action, self.non_spatial_action, self.value = net

      # Set targets and masks
      self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
      self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.image_size**2], name='spatial_action_selected')
      self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, self.isize], name='valid_non_spatial_action')
      self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, self.isize], name='non_spatial_action_selected')
      self.value_target = tf.placeholder(tf.float32, [None], name='value_target')

      # Compute log probability
      spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
      spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
      non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
      valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action, axis=1)
      valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
      non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
      non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
      self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))
      self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))

      # Compute losses, more details in https://arxiv.org/abs/1602.01783
      # Policy loss and value loss
      action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
      advantage = tf.stop_gradient(self.value_target - self.value)
      policy_loss = - tf.reduce_mean(action_log_prob * advantage)
      value_loss = - tf.reduce_mean(self.value * advantage)
      self.summary.append(tf.summary.scalar('policy_loss', policy_loss))
      self.summary.append(tf.summary.scalar('value_loss', value_loss))

      self.valueTensor = value_loss
      self.policyTensor = policy_loss

      # TODO: policy penalty
      loss = policy_loss + value_loss

      # Build the optimizer
      self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
      #TODO: Play around with variables and change the RMSPRopOptimizer. (which ones perform best)
      #opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
      opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-10)

      grads = opt.compute_gradients(loss)
      cliped_grad = []
      for grad, var in grads:
        self.summary.append(tf.summary.histogram(var.op.name, var))
        self.summary.append(tf.summary.histogram(var.op.name+'/grad', grad))
        grad = tf.clip_by_norm(grad, 10.0)
        cliped_grad.append([grad, var])
      self.train_op = opt.apply_gradients(cliped_grad)
      self.summary_op = tf.summary.merge(self.summary)

      self.saver = tf.train.Saver(max_to_keep=100)


  def step(self, obs, wrapper):
    screen, minimap, info = wrapper.generate_network_input()

    feed = {self.minimap: minimap,
            self.screen: screen,
            self.info: info}
    non_spatial_action, spatial_action = self.sess.run(
      [self.non_spatial_action, self.spatial_action],
      feed_dict=feed)

    # #TODO - Select an action and a spatial target
    # non_spatial_action = non_spatial_action.ravel()
    # spatial_action = spatial_action.ravel()
    # valid_actions = obs.observation['available_actions']
    # act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
    # target = np.argmax(spatial_action)
    #
    # target = [int(target // self.ssize), int(target % self.ssize)]

    # #TODO - Epsilon greedy exploration
    # if self.training and np.random.rand() < self.epsilon[0]:
    #   act_id = np.random.choice(valid_actions)
    # if self.training and np.random.rand() < self.epsilon[1]:
    #   dy = np.random.randint(-4, 5)
    #   target[0] = int(max(0, min(self.ssize-1, target[0]+dy)))
    #   dx = np.random.randint(-4, 5)
    #   target[1] = int(max(0, min(self.ssize-1, target[1]+dx)))

    # #TODO - Set act_id and act_args (possibly Not needed for DRDC Simulator)
    # act_args = []
    # for arg in actions.FUNCTIONS[act_id].args:
    #   if arg.name in ('screen', 'minimap', 'screen2'):
    #     act_args.append([target[1], target[0]])
    #   else:
    #     act_args.append([0])  # TODO: Be careful
    # return actions.FunctionCall(act_id, act_args)


  def update(self, rbs, disc, lr, cter):
    # Compute R, which is value of the last observation
    obs = rbs[-1][-1]
    if obs.last():
      R = 0
    else:
      minimap = np.array(obs.observation['minimap'], dtype=np.float32)
      minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
      screen = np.array(obs.observation['screen'], dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, obs.observation['available_actions']] = 1

      feed = {self.minimap: minimap,
              self.screen: screen,
              self.info: info}
      R = self.sess.run(self.value, feed_dict=feed)[0]

    # Compute targets and masks
    minimaps = []
    screens = []
    infos = []

    self.R = R

    value_target = np.zeros([len(rbs)], dtype=np.float32)
    value_target[-1] = R

    valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)
    spatial_action_selected = np.zeros([len(rbs), self.image_size**2], dtype=np.float32)
    valid_non_spatial_action = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)
    non_spatial_action_selected = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)

    rbs.reverse()
    for i, [obs, action, next_obs] in enumerate(rbs):
      minimap = np.array(obs.observation['minimap'], dtype=np.float32)
      minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
      screen = np.array(obs.observation['screen'], dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, obs.observation['available_actions']] = 1

      minimaps.append(minimap)
      screens.append(screen)
      infos.append(info)

      reward = obs.reward
      act_id = action.function
      act_args = action.arguments

      value_target[i] = reward + disc * value_target[i-1]

      valid_actions = obs.observation["available_actions"]
      valid_non_spatial_action[i, valid_actions] = 1
      non_spatial_action_selected[i, act_id] = 1

      args = actions.FUNCTIONS[act_id].args
      for arg, act_arg in zip(args, act_args):
        if arg.name in ('screen', 'minimap', 'screen2'):
          ind = act_arg[1] * self.image_size + act_arg[0]
          valid_spatial_action[i] = 1
          spatial_action_selected[i, ind] = 1

    minimaps = np.concatenate(minimaps, axis=0)
    screens = np.concatenate(screens, axis=0)
    infos = np.concatenate(infos, axis=0)

    # Train
    feed = {self.minimap: minimaps,
            self.screen: screens,
            self.info: infos,
            self.value_target: value_target,
            self.valid_spatial_action: valid_spatial_action,
            self.spatial_action_selected: spatial_action_selected,
            self.valid_non_spatial_action: valid_non_spatial_action,
            self.non_spatial_action_selected: non_spatial_action_selected,
            self.learning_rate: lr}
    _, summary = self.sess.run([self.train_op, self.summary_op], feed_dict=feed)
    self.summary_writer.add_summary(summary, cter)

    #PolicyLoss and ValueLoss tensor grabbed from build model. (dtype = float32)
    policyLoss, valueLoss = self.sess.run([self.policyTensor, self.valueTensor], feed_dict=feed)

    self.policyLoss = policyLoss
    self.valueLoss = valueLoss



  def save_model(self, path, count):
    self.saver.save(self.sess, path+'/model.pkl', count)


  def load_model(self, path):
    ckpt = tf.train.get_checkpoint_state(path)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    return int(ckpt.model_checkpoint_path.split('-')[-1])