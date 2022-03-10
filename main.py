import sys
import os

import numpy
import numpy as np
from utils import detection_pairs, decomposed_velocity, pos_to_dist, sort
from random import random, randint
import collections
import pyglet
import time
import wrapper
import tensorflow as tf
import a3c_agent


# Uncomment below for testing purposes (consistent initialization)
# np.random.seed(0)
# random.seed(0)

#Printing without truncation
numpy.set_printoptions(threshold=sys.maxsize)

class Features:
    """
    Description:
        This class is used to define the main features that will be present in the simulation.
    Attributes:
        target_count : int
            The number of targets.
        defender_count : int
            The number of defenders (interceptor missile launchers).
        targets : numpy.ndarray
            shape : target_count, 3
            datatype : numpy.float64
            Stores all information pertaining to the targets. Each row represents a target, and each column represents
            information pertaining to the target as follows:
                0 : x-position
                1 : y-position
                2 : value
        defenders : numpy.ndarray
            shape : defender_count, 4
            datatype : numpy.float64
            Stores all information pertaining to the defenders. Each row represents a defender, and each column
            represents information pertaining to the defenders as follows:
                0 : x-position
                1 : y-position
                2 : ammunition
                3 : reload time
        cml_ammunition : numpy.ndarray
            shape : 1,
            datatype : np.int32
            Stores the starting ammunition of the cruise missile launcher (attacker).
        cml : numpy.ndarray
            shape : 1, 4
            datatype : numpy.float64
            Stores all information pertaining to the cruise missile launcher (attacker's launcher). Since there is just
            one launcher, there is just one row representing this launcher. Each column represents information
            pertaining to the launcher as follows:
                0 : x-position
                1 : y-position
                2 : ammunition
                3 : reload time
        attackers : numpy.ndarray
            shape : 10, 7
            datatype : numpy.float64
            Stores all information pertaining to the attackers (missiles). Each row represents an attacker (missile),
            and each column represents information pertaining to the attackers (missiles) as follows:
                0 : x-position
                1 : y-position
                2 : target index
                3 : x-velocity
                4 : y-velocity
                5 : time steps to hit
                6 : valid/active
    """
    def __init__(self, defender_count, target_count):
        """
        input parameters:
            defender_count : int
            target_count : int
        """
        self.target_count = target_count
        self.defender_count = defender_count
        # Targets: Position (x,y), Value (50-150->rand), Hit (0,1->0)
        targets_x = np.random.uniform(low=0, high=500, size=(target_count, 1))  # On left side of map
        targets_y = np.random.uniform(low=0, high=1000, size=(target_count, 1))
        targets_value = np.random.uniform(low=50, high=150, size=(target_count, 1))
        self.targets = np.column_stack((targets_x, targets_y, targets_value))

        # Interceptors: Position (x,y), Ammunition (5-10->rand), Reload Delay (0-5->0),
        defenders_x = np.random.uniform(low=0, high=500, size=(defender_count, 1))  # On left side of map (this placement will need to be modified through a policy)
        defenders_y = np.random.uniform(low=0, high=1000, size=(defender_count, 1))
        defenders_ammunition = np.random.randint(low=5, high=10, size=(defender_count, 1))
        defenders_reload_delay = np.zeros(shape=(defender_count, 1))
        self.defenders = np.column_stack((defenders_x, defenders_y, defenders_ammunition, defenders_reload_delay))

        # Cruise Missile Launcher (cml): Position (x,y), Ammunition (5-10->rand), Reload Delay (0-5->0)
        cml_x = np.random.uniform(500, 1000, 1)  # On right side of map
        cml_y = np.random.uniform(0, 1000, 1)
        cml_ammunition = np.random.randint(5, 10, 1)
        self.cml_ammunition = cml_ammunition  # Setting a separate variable to track starting ammunition as it will be utilized in the launch function

        cml_reload_delay = np.zeros(1)
        self.cml = np.column_stack((cml_x, cml_y, cml_ammunition, cml_reload_delay))

        # Attackers (cruise missiles): Position (x,y), Target Index, Velocity x, Velocity y, time steps to hit,
        # Valid (0,1->0)
        attackers_x = np.full(10, cml_x)
        attackers_y = np.full(10, cml_y)
        attackers_dest_vel_val = np.zeros((10, 5))  # Initializing destination (target index), velocity
        # (x, y components), time steps to hit, validity as all zero in single shot
        self.attackers = np.column_stack((attackers_x, attackers_y, attackers_dest_vel_val))




class Simulation:
    """
    Description:
        This class is used to define the simulation environment and methods.
    Class Attributes:
        DEFENDER_DETECTION_RADIUS : float
            The range in km within which defenders are able to detect attackers.
        DEFENDER_INTERCEPTION_RADIUS : float
            The range in km within which defenders are able to intercept attackers.
        ATTACKER_RELOAD_DELAY : float
            The time in minutes required between successive attacker missile launches.
        DEFENDER_RELOAD_DELAY : float
            The time in minutes required between successive defender interception launches.
        ATTACKER_MISSILE_VELOCITY : float
            The velocity in km/minute of an attacker's missile.
        DEFENDER_MISSILE_VELOCITY : float
            The velocity in km/minute of a defender's missile.
    Instance Attributes:
        current_time : float
            The time elapsed in minutes since the start of the simulation.
        time_step_size : float
            The time elapsed in minutes through each step in the simulation.
        target_count : int
            The number of targets.
        defender_count : int
            The number of defenders (interceptor missile launchers).
        features : class Features
            The features present in the simulation (refer to feature class specifications).
        detections : np.ndarray
            shape : self.features.defender_count, 10
            datatype : np.int32
            A numpy array used for masking opportunities by detections. Each row represents a defender and each column
            represents a possible missile.
        defender_opportunities : np.ndarray
            shape : self.features.defender_count, 10, 8
            datatype : np.float64
            A numpy array used for storing opportunities. There are three dimensions which are specified as follows:
                dim0 : represents each defender
                dim1 : represents each attacker (missile)
                dim2 : represents information pertaining to opportunity (between corresponding dim0/dim1)
                    0 : opportunity 1 mask
                    1 : opportunity 1 time step of interception
                    2 : opportunity 1 x-coordinate of interception
                    3 : opportunity 1 y-coordinate of interception
                    4 : opportunity 2 mask
                    5 : opportunity 2 time step of interception
                    6 : opportunity 2 x-coordinate of interception
                    7 : opportunity 2 y-coordinate of interception
        interception_event_list : list
            A list used to store the interception opportunities taken, and corresponding interception information.
            Updated in each step of the simulation to remove events that have been completed or invalidated, and to add
            new opportunities taken. The sub-lists of this list, representing opportunities taken, will be structured
            as follows:
                0 : time steps until event
                1 : x-coordinate of event
                2 : y-coordinate of event
                3 : defender index
                4 : attacker index
    """
    DEFENDER_DETECTION_RADIUS = 400.  # km
    DEFENDER_INTERCEPTION_RADIUS = 160.  # km
    ATTACKER_RELOAD_DELAY = 5.  # min
    DEFENDER_RELOAD_DELAY = 5.  # min
    ATTACKER_MISSILE_VELOCITY = 14.  # Assumed to be in units of km/min, with time step assumed in min
    DEFENDER_MISSILE_VELOCITY = 12.  # km/min

    def __init__(self, features, time_step_size, wrapper):
        """
        input parameters:
            features : class Features
                The features in the simulation environment.
            time_step_size : int
                The time step size in minutes.
        """
        self.current_time = 0
        self.time_step_size = time_step_size
        self.features = features
        self.detections = np.zeros(shape=(self.features.defender_count, 10))
        self.defender_opportunities = np.zeros(shape=(self.features.defender_count, 10, 8))
        self.interception_event_list = []

        self.wrapper = wrapper

    def print_env(self):
        """
        Used to cleanly print the simulation information with relevant column titles.
        """
        np.set_printoptions(precision=8, linewidth=125, suppress=True)
        print('         CRUISE MISSILE LAUNCHER INFORMATION')
        print('  x_location  | y_location  | ammunition | reload_delay')
        print(self.features.cml)
        print('                 CRUISE MISSILE INFORMATION')
        print('   x_current  | y_current   |target_index| x_velocity | y_velocity | time_steps | active (0/1)')
        print(self.features.attackers)

        print('                INTERCEPTOR INFORMATION')
        print('  x_location  | y_location  | ammunition | reload_delay')
        print(self.features.defenders)

        print('             TARGET INFORMATION')
        print('  x_location  | y_location | value')
        print(self.features.targets)

    # This function initiates a new cruise missiles if the action is valid
    def cruise_missile_launch(self, target_idx):
        """
        Initialize a new attacker (cruise missile). If the action is valid, the relevant row in self.features.attackers
        is updated with all relevant parameters.
        input parameters:
            target_idx : int
                The target index that the missile is being launched towards.
        returns:
            True if launch was successful
            False if launch failed
        """

        # Converting to int from numpy.ndarray
        ammunition_starting = int(self.features.cml_ammunition)
        ammunition_remaining = int(self.features.cml[:, 2])
        # Checking that target index is valid
        if target_idx > self.features.target_count:
            # Print used for testing
            print('Target index invalid, launch failed')
            return False
        # Ensuring ammunition still available and reload time is zero
        if ammunition_remaining == 0 or self.features.cml[:, 3] != 0:
            # Print used for testing
            print('launch failed, ammunition: ' + str(ammunition_remaining) + ' reload time: ' +
                  str(self.features.cml[:, 3]))
            return False

        # Making the missile destination the target index (and selecting the row based on total ammo - current ammo)
        self.features.attackers[ammunition_starting - ammunition_remaining, 2] = target_idx
        # Decrementing ammunition
        self.features.cml[:, 2] -= 1
        # Setting reload time to CML_RELOAD_DELAY in time steps
        self.features.cml[:, 3] = self.ATTACKER_RELOAD_DELAY/self.time_step_size
        # Setting Valid to 1
        self.features.attackers[ammunition_starting - ammunition_remaining, 6] = 1
        # Getting the origin x, y coordinates
        origin = self.features.cml[0, 0:2]
        # Getting the destination x, y coordinates
        destination = self.features.targets[target_idx, 0:2]
        # Setting the x and y components of velocity
        self.features.attackers[ammunition_starting - ammunition_remaining, 3:6] = decomposed_velocity(
            self.ATTACKER_MISSILE_VELOCITY, origin, destination, self.time_step_size)
        return True

    def update_detections(self):
        """
        Generates the detections mask, self.detections, by checking for each attacker/defender pair whether
        the attacker is within the defender's detection radius.
        """
        # We need to check for each interceptor and missile combination, whether the missile is within detection range
        self.detections = np.zeros(shape=(self.features.defender_count, 10))
        interceptors = self.features.defenders
        missiles = self.features.attackers
        # Generating all interceptor/missile combinations - done prior to valid filtering to preserve indices
        interceptor_missile_pairs = detection_pairs(interceptors[:, :2], missiles[:, :2], self.DEFENDER_DETECTION_RADIUS)
        # Filtering by active missiles - this gives us a list of valid interceptor/missile pairs
        valid_detections = interceptor_missile_pairs[np.where(missiles[:, 6][interceptor_missile_pairs[:, 1]] == 1)]
        self.detections[valid_detections[:, 0], valid_detections[:, 1]] = 1

    def update_opportunities(self):
        """
        Updates the opportunities for the current time step based on the environment features. Checks the trajectories
        of attackers, and calculates where a defender may be able to intercept based on the posssible trajectories of
        each if fired in the current time step.
        """
        # Axy0 is now of dim: #-attackers x (x,y) x #-defenders. So, [:,:,i] gives the coords of all missiles shifted
        # to ith defender's coordinate frame
        Axy0 = self.features.attackers[:, 0:2, np.newaxis] - self.features.defenders[:, 0:2, np.newaxis].T
        Ax0 = Axy0[:, 0, :]
        Ay0 = Axy0[:, 1, :]
        # Since Avx and Avy are stored on a per timestep basis, need to convert to per minute basis
        Avx = self.features.attackers[:, 3]/self.time_step_size
        Avy = self.features.attackers[:, 4]/self.time_step_size

        a = Avx**2 + Avy**2 - self.DEFENDER_MISSILE_VELOCITY**2
        a = a[:, np.newaxis]
        b = 2*(Avx[:, np.newaxis]*Ax0 + Avy[:, np.newaxis]*Ay0)
        c = Ax0**2 + Ay0**2
        # t1 and t2 will now hold the interception opportunity times in minutes
        discriminant = b**2 - 4*a*c
        # Absolute value used to avoid runtime warning for negative discriminant, as we will mask by negative
        # discriminant afterwards anyways.
        t1 = (-b + np.sqrt(np.abs(discriminant))) / (2*a)
        t2 = (-b - np.sqrt(np.abs(discriminant))) / (2*a)

        t1x = t1*Avx[:, np.newaxis] + Ax0
        t1y = t1*Avy[:, np.newaxis] + Ay0
        t2x = t2*Avx[:, np.newaxis] + Ax0
        t2y = t2*Avy[:, np.newaxis] + Ay0

        # We also wish to store the t1 and t2 times in terms of time steps, done as follows
        t1_ts = t1/self.time_step_size
        t2_ts = t2/self.time_step_size

        # t1xy now holds the x,y coordinates of the t1 opportunities in the interceptors' coordinate frames
        t1xy = np.stack((t1x, t1y), axis=1)
        t2xy = np.stack((t2x, t2y), axis=1)

        # we also need to mask for those interceptors that are still reloading
        interceptor_ready = np.zeros(shape=(10, self.features.defender_count))
        interceptor_ready[:, np.where(self.features.defenders[:, 3] == 0)] = 1
        # t1xy_distances now holds the distance from each interceptor (columns) to each missile (row) at opportunity t1
        t1xy_distances = pos_to_dist(t1xy[:, 0, :], t1xy[:, 1, :])
        t2xy_distances = pos_to_dist(t2xy[:, 0, :], t2xy[:, 1, :])

        t1_interception_mask = np.zeros(shape=(10, self.features.defender_count))
        t2_interception_mask = np.zeros(shape=(10, self.features.defender_count))
        t1_interception_mask[np.where((t1xy_distances < self.DEFENDER_INTERCEPTION_RADIUS) & (t1xy_distances > 0) &
                                      (t1_ts > 0) & (discriminant > 0))] = 1
        t2_interception_mask[np.where((t2xy_distances < self.DEFENDER_INTERCEPTION_RADIUS) & (t2xy_distances > 0) &
                                      (t2_ts > 0) & (discriminant > 0))] = 1

        self.update_detections()
        op1_mask = t1_interception_mask*self.detections.T*interceptor_ready
        op2_mask = t2_interception_mask*self.detections.T*interceptor_ready
        # Shifting back to global coordinate frame
        t1xy_global = t1xy + self.features.defenders[:, 0:2].T
        t2xy_global = t2xy + self.features.defenders[:, 0:2].T
        # op1 and op2 now have shape #-missiles, (mask, x, y), #-interceptors
        op1 = np.concatenate((op1_mask[:, np.newaxis, :], t1_ts[:, np.newaxis, :], t1xy_global), axis=1)
        op1 = np.transpose(op1, (2, 0, 1))
        op2 = np.concatenate((op2_mask[:, np.newaxis, :], t2_ts[:, np.newaxis, :], t2xy_global), axis=1)
        op2 = np.transpose(op2, (2, 0, 1))
        self.defender_opportunities = np.concatenate((op1, op2), axis=2)

    def decrement_reload_times(self):
        """
        Decrementing reload time of the cruise missile launcher and each defender by 1 time step, while ensuring the
        reload time doesn't become negative.
        """
        self.features.cml[:, 3] -= 1
        # If wait time is less than zero, set it to zero
        self.features.cml[:, 3] = np.where(self.features.cml[:, 3] < 0, 0, self.features.cml[:, 3])
        self.features.defenders[:, 3] -= 1
        # If wait time is less than zero, set it to zero
        self.features.defenders[:, 3] = np.where(self.features.defenders[:, 3] < 0, 0, self.features.defenders[:, 3])

    def update_cruise_missile_positions(self):
        """
        Updating all valid attackers' positions by adding their distance/time step to their positions. Additionally,
        the time steps until they hit their targets are decremented by 1.
        """
        # We update the positions of all cruise missiles (using cms for more condensed code)
        cms = self.features.attackers
        # Checking that at least 1 attacker is valid.
        if cms[np.where(cms[:, 6] == 1)].size > 0:
            # updating active missiles based on decomposed velocity columns
            cms[np.where(cms[:, 6] == 1), :2] += cms[np.where(cms[:, 6] == 1), 3:5]
            # decrementing time steps to hit
            cms[np.where(cms[:, 6] == 1), 5] -= 1
        self.features.attackers = cms

    # This is where a policy for selecting actions can be inserted - for now we are just extracting the actions
    # possible and taking a random action for each missile
    def defender_actions(self):
        """
        Selecting an action for each defender from valid actions, if available. Note that it isn't a proper random
        selection right now as first options are favored.
        returns:
            None if no valid actions available
            actions :  list
                A list of actions with their corresponding information. The sub-lists in this list contain:
                    0 : time steps until action completion
                    1 : x-coordinate of event
                    2 : y-coordinate of event
                    3 : defender index
                    4 : attacker index
                    5 : probability of success
        """
        actions = []
        pairs1 = np.where(self.defender_opportunities[:, :, 0] == 1)
        pairs2 = np.where(self.defender_opportunities[:, :, 4] == 1)
        options1 = collections.defaultdict(list)
        options2 = collections.defaultdict(list)
        if len(pairs1[0]) == 0 and len(pairs2[0]) == 0:
            return None
        else:
            for i in range(len(pairs1[0])):
                options1[pairs1[0][i]].append(pairs1[1][i])
            for i in range(len(pairs2[0])):
                options2[pairs2[0][i]].append(pairs2[1][i])
        for i in range(self.features.defender_count):
            # If no options for the interceptor, or reload time is not zero, or no ammunition remaining, continue
            if len(options1[i]) == 0 or self.features.defenders[i, 3] > 0 or self.features.defenders[i, 2] == 0:
                continue
            else:
                action_idx = randint(1, len(options1[i])) - 1
                missile_index = options1[i][action_idx]
                interceptor_index = i
                time_step = self.defender_opportunities[interceptor_index, missile_index, 1]
                x = self.defender_opportunities[interceptor_index, missile_index, 2]
                y = self.defender_opportunities[interceptor_index, missile_index, 3]
                actions.append([time_step, x, y, interceptor_index, missile_index, 0.7])
                self.features.defenders[interceptor_index, 3] = self.DEFENDER_RELOAD_DELAY / self.time_step_size
                self.features.defenders[interceptor_index, 2] -= 1
        for i in range(self.features.defender_count):
            if len(options2[i]) == 0 or self.features.defenders[i, 3] > 0 or self.features.defenders[i, 2] == 0:
                continue
            else:
                action_idx = randint(1, len(options2[i])) - 1
                missile_index = options2[i][action_idx]
                interceptor_index = i
                time_step = self.defender_opportunities[interceptor_index, missile_index, 5]
                x = self.defender_opportunities[interceptor_index, missile_index, 6]
                y = self.defender_opportunities[interceptor_index, missile_index, 7]
                actions.append([time_step, x, y, interceptor_index, missile_index, 0.7])
                self.features.defenders[interceptor_index, 3] = self.DEFENDER_RELOAD_DELAY / self.time_step_size
                self.features.defenders[interceptor_index, 2] -= 1

        return actions

    # This is where a policy for attacker actions could be inserted, for now it is random
    def attacker_actions(self):
        """
        Attacker to take a random action if valid.
        return:
            -1 if no ammunition remaining or still reloading
            random integer corresponding to a target index otherwise
        """
        if self.features.cml[:, 2] == 0 or self.features.cml[:, 3] != 0:
            return -1
        else:
            return randint(1, self.features.target_count) - 1

    def decrement_interception_event_list(self):
        """
        Decrement the events in the interception_event_list by 1 time step.
        """
        for element in self.interception_event_list:
            element[0] -= 1

    def add_interception_events(self, action_list):
        """
        Used to add the defender actions to self.interception_event_list, and set reload time for each defender that
        takes an action.
        input parameters:
            action_list : list
                A list of lists, where the sub-lists contain information about the interception event as follows:
                    0 : time steps until action completion
                    1 : x-coordinate of event
                    2 : y-coordinate of event
                    3 : defender index
                    4 : attacker index
                    5 : probability of success
        """
        if action_list:
            # Combining old and new lists into a single list
            self.interception_event_list.extend(action_list)


    def process_events(self):
        """
        Generates a unified event list of interception events in self.interception_event_list and missile impact events
        based on the active attackers (missiles) in the environment. This list is then sorted, and events with time step
        less than 1 are processed in order.
        """
        # Note: interception event defined as 0: time step, 1: x, 2:y, 3:interceptor_idx, 4:missile_idx, 5: p_success, 6: event_type (0)
        # For target hit event, will define as 0: time step, 1:x, 2:y, 3:target_idx,      4:missile_idx, 5: p_success, 6: event_type (1)
        event_list = []
        if len(self.interception_event_list) > 0:
            # adding a 'type' column so events can be handled appropriately in final list
            interception_type_fill = np.full(len(self.interception_event_list), 0)
            interception_events = np.column_stack((self.interception_event_list, interception_type_fill)).tolist()
            event_list.extend(interception_events)
        active_missile_indeces = np.where(self.features.attackers[:, 6] == 1)
        if len(active_missile_indeces) > 0:
            p_success_fill = np.full(len(active_missile_indeces[0]), 0.7)
            hit_type_fill = np.full(len(active_missile_indeces[0]), 1)
            active_missiles = self.features.attackers[active_missile_indeces]
            hit_events = np.column_stack((active_missiles[:, [5, 0, 1, 2]], active_missile_indeces[0], p_success_fill, hit_type_fill)).tolist()
            event_list.extend(hit_events)

        sort(event_list)
        if len(event_list) == 0:
            return
        else:
            current_event = event_list[0]
        # since stored in time steps, check if occurs in current time step
        while current_event[0] < 1:
            event_type = current_event[6]  # 1 for attacker hit target, 0 for defender hit missile
            # if p_success greater than random number generated, missile is destroyed
            indeces_for_removal_event_list = []
            indeces_for_removal_interception_event_list = []
            if random() < current_event[5]:
                if event_type == 0:  # if event is defender hitting missile
                    print('Current Time: ', self.current_time)
                    print(f'Missile {current_event[4]} destroyed by interceptor {current_event[3]}')
                    # setting validity of relevant attacker to 0 if successful
                    self.features.attackers[int(current_event[4]), 6] = 0
                elif event_type == 1:  # This elif isn't necessary as an elif per se but including for clarity
                    print('Current Time: ', self.current_time)
                    print(f'Missile {current_event[4]} hit target {current_event[3]}')
                    # A successful target hit decreases target value by multiplying by p_success
                    self.features.targets[int(current_event[3]), 2] *= current_event[5]
                    self.features.attackers[int(current_event[4]), 6] = 0

                # Identifying future events related to that missile and storing their indeces
                for idx, event in enumerate(event_list):
                    if int(event[4]) == int(current_event[4]):
                        indeces_for_removal_event_list.append(idx)
                for idx, event in enumerate(self.interception_event_list):
                    if int(event[4]) == int(current_event[4]):
                        indeces_for_removal_interception_event_list.append(idx)
            # Removing future invalidated events with last index first to preserve index order
            for index in sorted(indeces_for_removal_event_list, reverse=True):
                event_list.pop(index)
            for index in sorted(indeces_for_removal_interception_event_list, reverse=True):
                self.interception_event_list.pop(index)

            if len(event_list) == 0:
                return
            else:
                current_event = event_list[0]


    def step(self, defender_actions=None, attacker_action=-1):
        """
        The core of the simulator, the step function. Utilizes all previously discussed methods to update the
        environment by taking actions, moving movable features (attackers), and processing events.
        input parameters:
            defender_actions : list
                A list of defender actions taken based on the current environment in this time step.
            attacker_action : int
                The attacker action (target index).
        """
        # Updating time
        self.current_time += self.time_step_size
        # Decrementing attacker and defender reload times
        self.decrement_reload_times()

        # Decrementing interception event times
        self.decrement_interception_event_list()
        # Assuming empty attacker_action is equivalent to 'do nothing', otherwise action holds idx of target
        if attacker_action > -1:
            self.cruise_missile_launch(attacker_action)
        if defender_actions:
            self.add_interception_events(defender_actions)

        self.process_events()
        # Updating cruise_missile_positions
        self.update_cruise_missile_positions()
        # Updating the detections and opportunity list
        self.update_detections()
        self.update_opportunities()
        #update wrapper images
        self.wrapper.step_update(self.features, self.defender_opportunities)



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
        angle = np.arctan2(x_step, y_step)*180/np.pi
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
        circle1 = pyglet.shapes.Circle(x=x_current, y=y_current, radius=env.DEFENDER_DETECTION_RADIUS, color=(255, 225, 30))
        # circle1.opacity = 128
        circle2 = pyglet.shapes.Circle(x=x_current, y=y_current, radius=env.DEFENDER_INTERCEPTION_RADIUS, color=(255, 0, 0))
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
    x_current, y_current = env.features.cml[0, :2]
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
        pyglet.image.get_buffer_manager().get_color_buffer().save('./images/'+str(env.current_time)+'.png')
        # Used to slow down the simulation
        time.sleep(0.2)
        pyglet.app.exit()

def setup_env(targets, defenders):
    targets = targets
    defenders = defenders
    ts_size = 3  # minutes
    sim_length = 100  # minutes
    total_steps = int(sim_length/ts_size)

    DEFENDER_DETECTION_RADIUS = 400.  # km
    DEFENDER_INTERCEPTION_RADIUS = 160.  # km

    f = Features(target_count=targets, defender_count=defenders)

    # Initialize wrapper
    w = wrapper.Wrapper(target_count=targets, defender_count=defenders)
    w.init_input(f, DEFENDER_DETECTION_RADIUS, DEFENDER_INTERCEPTION_RADIUS)

    env = Simulation(features=f, time_step_size=ts_size, wrapper=w)

    window = pyglet.window.Window(width=1000, height=1000)
    pyglet.clock.schedule(visualize, env=env)
    return env


if __name__ == '__main__':
    targets = 8
    defenders = 6
    ts_size = 3  # minutes
    sim_length = 100  # minutes
    total_steps = int(sim_length/ts_size)

    DEFENDER_DETECTION_RADIUS = 400.  # km
    DEFENDER_INTERCEPTION_RADIUS = 160.  # km

    f = Features(target_count=targets, defender_count=defenders)

    # Initialize wrapper
    w = wrapper.Wrapper(target_count=targets, defender_count=defenders)
    w.init_input(f, DEFENDER_DETECTION_RADIUS, DEFENDER_INTERCEPTION_RADIUS)

    env = Simulation(features=f, time_step_size=ts_size, wrapper=w)

    window = pyglet.window.Window(width=1000, height=1000)

    pyglet.clock.schedule(visualize, env=env)
    for i in range(total_steps):
        env.step(attacker_action=env.attacker_actions(), defender_actions=env.defender_actions())
        # Comment/Uncomment to render simulation
        pyglet.clock.tick()
        pyglet.app.run()


