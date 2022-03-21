from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

#Used for PySc2 environment
import enum
import collections

import numpy
import numpy as np


def run_loop(agents, env, max_frames=0):
  """A run loop to have agents and an environment interact."""
  start_time = time.time()

  wrapper = env.return_wrapper()

  try:
    while True:
      num_frames = 0
      timesteps = wrapper.reset(env)
      for a in agents:
        a.reset()
      while True:
        num_frames += 1
        last_timesteps = timesteps
        actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
        timesteps = wrapper.step(env, env.random_defender_actions(env.features.defender_count, env.features.attacker_launcher_ammunition), (env.process_attacker_actions(), actions))
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



