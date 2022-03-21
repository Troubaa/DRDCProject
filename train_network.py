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

from run_loop import run_loop
from main import MissileEnv
from main import Features
from wrapper import Wrapper

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
flags.DEFINE_integer("max_steps", 10, "Total steps for training.")
flags.DEFINE_integer("snapshot_step", int(1e3), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_string("device", "0", "Device for training.")

flags.DEFINE_string("log_name", "DRDCSimulator", "Name of a simulator used (Different versions/updates made).")
flags.DEFINE_bool("render", True, "Whether to render with visuals.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "a3c_agent.A3CAgent", "Which agent to run.")
flags.DEFINE_string("net", "fcn", "atari or fcn.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(), "Bot's strength.")
flags.DEFINE_integer("max_agent_steps", 50, "Total agent steps.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

flags.DEFINE_integer("targets", 8, "Number of targets (should not change)")
flags.DEFINE_integer("defenders", 6, "Number of defenders (should not change)")
flags.DEFINE_integer("ts_size", 5, "Minutes")


FLAGS(sys.argv)

if FLAGS.training:
  PARALLEL = FLAGS.parallel
  MAX_AGENT_STEPS = FLAGS.max_agent_steps
  DEVICE = ['/gpu:'+dev for dev in FLAGS.device.split(',')]
else:
  PARALLEL = 1
  MAX_AGENT_STEPS = 1e5
  DEVICE = ['/cpu:0']

LOG = FLAGS.log_path+FLAGS.log_name+'/'+FLAGS.net
SNAPSHOT = FLAGS.snapshot_path+FLAGS.log_name+'/'+FLAGS.net
if not os.path.exists(LOG):
  os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
  os.makedirs(SNAPSHOT)


def run_thread(agent, render):
  with MissileEnv(
    features=Features(target_count=FLAGS.targets, defender_count=FLAGS.defenders), time_step_size=FLAGS.ts_size,
          wrapper=Wrapper(target_count=FLAGS.targets, defender_count=FLAGS.defenders, discount=FLAGS.discount)) as env:
    #env = available_actions_printer.AvailableActionsPrinter(env)

    # Only for a single player!
    replay_buffer = []
    for recorder, is_done in run_loop([agent], env, render, MAX_AGENT_STEPS):
      if FLAGS.training:
        replay_buffer.append(recorder)
        if is_done:
          counter = 0
          with LOCK:
            global COUNTER
            COUNTER += 1
            counter = COUNTER
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

  # Setup agents
  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)


  agents = []
  for i in range(PARALLEL):
    agent = agent_cls(FLAGS.training, FLAGS.minimap_resolution, FLAGS.screen_resolution)
    agent.build_model(i > 0, DEVICE[i % len(DEVICE)], FLAGS.net)
    agents.append(agent)

  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  summary_writer = tf.summary.FileWriter(LOG)
  for i in range(PARALLEL):
    agents[i].setup(sess, summary_writer)

  agent.initialize()
  if not FLAGS.training or FLAGS.continuation:
    global COUNTER
    COUNTER = agent.load_model(SNAPSHOT)

  # Run threads
  threads = []
  for i in range(PARALLEL - 1):
    t = threading.Thread(target=run_thread, args=(agents[i], FLAGS.log_name, False))
    threads.append(t)
    t.daemon = True
    t.start()
    time.sleep(5)

  run_thread(agents[-1], FLAGS.render)

  for t in threads:
    t.join()

  if FLAGS.profile:
    print(stopwatch.sw)


if __name__ == "__main__":
  app.run(_main)
