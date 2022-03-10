from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import importlib
import threading

from absl import app
from absl import flags
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
import tensorflow as tf
import a3c_agent
import main as sim

SCORE = []
POLICY = []
VALUE = []

COUNTER = 0
LOCK = threading.Lock()
FLAGS = flags.FLAGS
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 0.0005, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("max_steps", 360, "Total steps for training.")
flags.DEFINE_integer("snapshot_step", int(1e3), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_string("device", "0", "Device for training.")

flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("image_size", 64, "Resolution for feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("targets", 8, "Number of Targets.")
flags.DEFINE_integer("defenders", 6, "Number of Targets.")

flags.DEFINE_string("agent", "agents.a3c_agent.A3CAgent", "Which agent to run.")
flags.DEFINE_string("net", "fcn", "fcn.")
flags.DEFINE_integer("max_agent_steps", 360, "Total agent steps.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

FLAGS(sys.argv)

if FLAGS.training:
  MAX_AGENT_STEPS = FLAGS.max_agent_steps
  DEVICE = ['/gpu:'+dev for dev in FLAGS.device.split(',')]
else:
  MAX_AGENT_STEPS = 1e5
  DEVICE = ['/cpu:0']

LOG = FLAGS.log_path+'/'+FLAGS.net
SNAPSHOT = FLAGS.snapshot_path+'/'+FLAGS.net
if not os.path.exists(LOG):
  os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
  os.makedirs(SNAPSHOT)


def run_loop(agent, env, max_frames=0):
  """A run loop to have agents and an environment interact."""
  start_time = time.time()

  try:
    while True:
      num_frames = 0
      timesteps = env.reset()
      agent.reset()
      while True:
        num_frames += 1
        last_timesteps = timesteps
        actions = [agent.step(timestep) for agent, timestep in zip(agent, timesteps)]
        timesteps = env.step(actions)
        # Only for a single player!
        is_done = (num_frames >= max_frames) or timesteps[0].last()
        yield [last_timesteps[0], actions[0], timesteps[0]], is_done
        if is_done:
          break
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)


def run_agent(agent):
    env = sim.setup_env(FLAGS.targets, FLAGS.defenders)
    # Only for a single player!
    replay_buffer = []
    counter = 0
    for recorder, is_done in run_loop(agent, env, MAX_AGENT_STEPS):
      if FLAGS.training:
        replay_buffer.append(recorder)
        if is_done:
          counter += 1
          # Learning rate schedule
          learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
          agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)
          replay_buffer = []

          obs = recorder[-1].observation
          score = obs["score_cumulative"][0]

          #Obtain Score, Policy Loss and Value Loss to determine that the Neural Network does learn.
          with LOCK:
            global POLICY
            global VALUE
            global SCORE
            policyLoss, valueLoss = agent.returngraphvalue()

            POLICY.append(policyLoss)
            VALUE.append(valueLoss)
            SCORE.append(int(score))
            policy = POLICY
            value = VALUE
            score = SCORE

          #If save data when we reach the snapshot step just in case program crashes half way through we still have some data.
          if counter % FLAGS.snapshot_step == 1:
            agent.save_model(SNAPSHOT, counter)

            policyFile = open("./Data/policydata.txt.", "a")
            valueFile = open("./Data/valuedata.txt.", "a")
            scoreFile = open("./Data/scoredata.txt.", "a")

            strPolicy = listToString(policy)
            strValue = listToString(value)
            strScore = listToString(score)

            policyFile.write('[')
            valueFile.write('[')
            scoreFile.write('[')

            policyFile.write(strPolicy)
            valueFile.write(strValue)
            scoreFile.write(strScore)

            policyFile.write(']\n')
            valueFile.write(']\n')
            scoreFile.write(']\n')

            policyFile.close()
            valueFile.close()
            scoreFile.close()

            policy.clear()
            value.clear()
            score.clear()

          #save the last snapshot of data before game closes.
          if counter >= FLAGS.max_steps:

            policyFile = open("./Data/policydata.txt.", "a")
            valueFile = open("./Data/valuedata.txt.", "a")
            scoreFile = open("./Data/scoredata.txt.", "a")

            strPolicy = listToString(policy)
            strValue = listToString(value)
            strScore = listToString(score)

            policyFile.write('[')
            valueFile.write('[')
            scoreFile.write('[')

            policyFile.write(strPolicy)
            valueFile.write(strValue)
            scoreFile.write(strScore)

            policyFile.write(']\n')
            valueFile.write(']\n')
            scoreFile.write(']\n')

            policyFile.close()
            valueFile.close()
            scoreFile.close()

            break

      elif is_done:
        obs = recorder[-1].observation
        score = obs["score_cumulative"][0]
        print('Your score is '+str(score)+'!')


    if FLAGS.save_replay:
      env.save_replay(agent.name)


# Function to convert
def listToString(s):
  # initialize an empty string
  str1 = ""

  i = 0
  # traverse in the string
  for ele in s:
    i += 1
    str1 += str(ele)
    if i != len(s):
      str1 += ", "

    # return string
  return str1

def _main(unused_argv):
  """Run agents"""
  stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
  stopwatch.sw.trace = FLAGS.trace

  agent = a3c_agent.A3CAgent(FLAGS.training, FLAGS.image_size, FLAGS.targets)
  agent.build_model(DEVICE[0 % len(DEVICE)])

  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  summary_writer = tf.summary.FileWriter(LOG)

  agent.setup(sess, summary_writer)

  agent.initialize()
  if not FLAGS.training or FLAGS.continuation:
    global COUNTER
    COUNTER = agent.load_model(SNAPSHOT)

  run_agent(agent)

  if FLAGS.profile:
    print(stopwatch.sw)


if __name__ == "__main__":
  app.run(_main)
