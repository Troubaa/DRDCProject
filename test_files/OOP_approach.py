from random import uniform, randrange
import numpy as np
from utils import euclidean_distance, update_position


class CruiseMissileLauncher:
    POSITION = [uniform(500, 1000), uniform(500, 1000)]  # Initial position, to be modified with function that considers physical limitations of map - for now just ensuring it is on the right
    RANGE = 800  # kilometers
    RELOAD_DELAY = 5  # minutes

    def __init__(self):
        self.ammunition = randrange(5, 10)  # Initial ammunition
        self.wait_time_left = 0  # Assuming ready to fire at start of simulation
        self.missiles_launched = []

    def launch_missile(self, targets_in_range):
        #targets_in_range coming from Simulation
        # TODO ___ need a policy for selecting the target from those in range
        target = targets_in_range[0]
        self.missiles_launched.append(CruiseMissile(self.POSITION, target))
        self.ammunition -= 1
        self.wait_time_left = self.RELOAD_DELAY

    def step(self, time_step):
        self.wait_time_left -= time_step # TODO combine into single simulation step function

    def get_info(self):
        # Function will term parameters relevant to the encoding of the problem
        return

class CruiseMissile:
    VELOCITY = 13.5  # km/min (weird to use but aligns with other units used)

    def __init__(self, position, target):
        self.position = position  # Position initialized to position of launcher
        self.TARGET = target  # target is constant for missile and selected by launcher class
        self.interception_probability = 0.  # Interception probability begins at zero

    def step(self, time_step):
        self.position = update_position(self.position, self.TARGET, self.VELOCITY, time_step) # TODO these step fns all should be combined into the simulation step function

    def get_info(self):
        # Function will return parameters relevant to the encoding of the problem
        return

class InterceptorMissileLauncher:
    POSITION = [uniform(0, 500), uniform(0, 500)]  # Initial position, to be modified with function that considers physical limitations of map - for now just ensuring it is on the left
    RELOAD_DELAY = 1  # minutes
    RANGE_INTERCEPTION = 160  # km
    RANGE_DETECTION = 300  # km, redundant for now if using overleaf definitions but including for future consideration

    def __init__(self):
        self.ammunition = randrange(5, 10)  # Initial ammunition
        self.wait_time_left = 0  # Assuming ready to fire at start of simulation
        self.action_list = []

    def generate_opportunities(self, cruise_missile_launcher, time_step):
        #variables required: self.RANGE_INTERCEPTION, self.RANGE_DETECTION, cruise_missile_launcher.
        # Function where the opportunities will be generated - this needs to be modified to be continuous
        for cruise_missile in cruise_missile_launcher.missiles_launched:
            if euclidean_distance(cruise_missile.position, self.POSITION) <= self.RANGE_DETECTION:
                # Possible concern here - the missile could in pass over 'detection' region of interceptor missile launcher without being registered if timestep is too large
                self.missiles_detected.append(cruise_missile)
        for cruise_missile in self.missiles_detected:
            pass

        return

    def get_info(self):
        # Function will return parameters relevant to the encoding of the problem
        return

class Target:
    POSITION = [uniform(0, 500), uniform(0, 500)]  # Initial position, to be modified with function that considers physical limitations of map - for now just ensuring it is on the left

    def __init__(self):
        self.value = randrange(50, 150)  # Arbitrary random value assignment

    def hit_event(self, success_probability):
        self.value *= success_probability
        return

    def get_info(self):
        # Function will return parameters relevant to the encoding of the problem
        return

class Simulation:
    def __init__(self, interceptor_quantity, target_quantity, time_step):
        self.interceptor_list = []
        self.target_list = []
        self.time_step = time_step
        self.current_time = 0
        self.missile_list = []
        # Initializing defender assets
        for i in range(interceptor_quantity):  # Creating a list which stores each interceptor class instance
            self.interceptor_list.append(InterceptorMissileLauncher())
        for i in range(target_quantity):  # Creating a list which stores each target class instance
            self.target_list.append(Target())

        # Initializing attacker
        self.cruise_missile_launcher = CruiseMissileLauncher()
        # Initializing new variable here to avoid having to repeatedly call the class method, but keeping the method separate from init for ease of reading
        self.attacker_valid_targets = self.update_attacker_actions()

    def get_info(self):
        # Function will return parameters relevant to the encoding of the problem, by calling and encoding from all class get_info functions
        return

    # Filter the list of targets and return the list that can be hit by the attacker
    def update_attacker_actions(self):
        targets_in_range_list = []
        for item in self.target_list:
            if euclidean_distance(item.POSITION, self.cruise_missile_launcher.POSITION) <= self.cruise_missile_launcher.RANGE:
                targets_in_range_list.append(item)
        return targets_in_range_list

    def update_interceptor_actions(self):
        # based on a cruise missile launched, need to check where it first intersects each detection circle (if at all)
        # then, use this point in Karla's equations to determine interception opportunities
        # Finally, need to limit interception opportunities to those that occur within range of interceptor
        interceptor_action_list = []
        return interceptor_action_list

    def step(self):
        # Apply actions taken
        # update time and positions
        #regenerate opportunities and actions
        action = self.attacker_valid_targets
        # At what stage is the value deducted from the attacking missile? Timestep of hit or timestep of its selection as a target?

        # Update positions of attacker missiles

        # Update which are detected by defenders (generate list of opportunities)


        self.current_time += self.time_step_size  # Increment the time


