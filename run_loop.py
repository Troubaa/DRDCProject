from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

#Used for PySc2 environment
import enum
import collections

import numpy
import numpy as np
import pyglet
import main

window = pyglet.window.Window(width=1000, height=1000)

def visualize(dt, env):
  """
  Generates the visual representation of the simulation using pyglet.
  input parameters:
      dt : IGNORE - required by API by not utilized in our case.
      env : The environment, an instance of Simulation class
  """
  interception_events = env.interception_event_list
  interceptions = []
  for interception in interception_events:
    x_pos, y_pos = interception[1:3]
    star = pyglet.shapes.Star(x=x_pos, y=y_pos, inner_radius=3, outer_radius=8, num_spikes=5, color=(0, 0, 255))

    interceptions.append(star)

  active_missiles = env.features.attackers[np.where(env.features.attackers[:, 6] == 1)]
  sprites = []

  missile_image = pyglet.image.load('pyglet_sprite_images/missile.png')
  missile_image.anchor_x = missile_image.width // 2
  missile_image.anchor_y = missile_image.height // 2
  for missile in active_missiles:
    x_current, y_current = missile[0:2]
    x_step, y_step = missile[3:5]
    angle = np.arctan2(x_step, y_step) * 180 / np.pi
    sprite = pyglet.sprite.Sprite(missile_image, x=x_current, y=y_current)
    sprite.rotation = angle
    sprite.scale = 0.025
    sprites.append(sprite)
  targets = []
  for target in env.features.targets:
    x_current, y_current = target[0:2]
    circle = pyglet.shapes.Circle(x=x_current, y=y_current, radius=5, color=(50, 225, 30))
    targets.append(circle)
  defender_detection = []
  defender_interception = []
  defenders = []
  labels = []
  for idx, defender in enumerate(env.features.defenders):
    x_current, y_current = defender[0:2]
    circle1 = pyglet.shapes.Circle(x=x_current, y=y_current, radius=env.DEFENDER_DETECTION_RADIUS,
                                   color=(255, 225, 30))
    # circle1.opacity = 128
    circle2 = pyglet.shapes.Circle(x=x_current, y=y_current, radius=env.DEFENDER_INTERCEPTION_RADIUS,
                                   color=(255, 0, 0))
    # circle2.opacity = 128
    circle3 = pyglet.shapes.Circle(x=x_current, y=y_current, radius=5, color=(255, 255, 255))
    label_text = str(idx)
    label = pyglet.text.Label(label_text,
                              font_name='Times New Roman',
                              font_size=36,
                              x=x_current, y=y_current,
                              anchor_x='center', anchor_y='center')
    defender_detection.append(circle1)
    defender_interception.append(circle2)
    # defenders.append(circle3)
    labels.append(label)

  boat_image = pyglet.image.load('pyglet_sprite_images/boat.png')
  x_current, y_current = env.features.attacker_launcher[0, :2]
  attacker_sprite = pyglet.sprite.Sprite(boat_image, x=x_current, y=y_current)
  attacker_sprite.scale = 0.08

  # Uncomment to render the map in background
  # background_image = pyglet.image.load('interest.PNG')
  # background_image.anchor_x = background_image.width // 2
  # background_image.anchor_y = background_image.height // 2

  @window.event
  def on_draw():
    window.clear()
    # background_image.blit(x=750, y=500, z=0)
    for radius in defender_detection:
      radius.draw()
    for radius in defender_interception:
      radius.draw()
    for defender in defenders:
      defender.draw()
    for label in labels:
      label.draw()
    for target in targets:
      target.draw()
    for sprite in sprites:
      sprite.draw()
    for interception in interceptions:
      interception.draw()
    attacker_sprite.draw()
    pyglet.image.get_buffer_manager().get_color_buffer().save('./images/' + str(env.current_time) + '.png')
    # Used to slow down the simulation
    # time.sleep(0.2)
    pyglet.app.exit()


def run_loop(agents, env, render, max_frames=0):
  """A run loop to have agents and an environment interact."""
  start_time = time.time()

  wrapper = env.return_wrapper()

  try:
    while True:
      num_frames = 0
      timesteps = wrapper.reset(env)

      wrapper = env.return_wrapper()

      if render:
        pyglet.clock.schedule(visualize, env=env)

      for a in agents:
        a.reset()
      while True:
        num_frames += 1
        last_timesteps = timesteps
        actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
        timesteps = wrapper.step(env, env.random_defender_actions(env.features.defender_count, env.features.attacker_launcher_ammunition), actions)

        #to render simulation
        if render:
          pyglet.clock.tick()
          pyglet.app.run()

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



