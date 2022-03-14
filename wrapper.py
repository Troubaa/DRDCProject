import numpy
import numpy as np
import enum
import collections
import tensorflow as tf
from network import build_fcn

"""
Author - Noah Sweetnam
2022-01-12
Wrapper.py - This file contains...
"""

class FeatureType(enum.Enum):
  SCALAR = 1
  CATEGORICAL = 2

class StepType(enum.IntEnum):
  """Defines the status of a `TimeStep` within a sequence."""
  # Denotes the first `TimeStep` in a sequence.
  FIRST = 0
  # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
  MID = 1
  # Denotes the last `TimeStep` in a sequence.
  LAST = 2

class ScreenFeatures(collections.namedtuple("ScreenFeatures", [
    "height_map", "visibility_map", "creep", "power", "player_id",
    "player_relative", "unit_type", "selected", "unit_hit_points",
    "unit_energy", "unit_shields", "unit_density", "unit_density_aa"])):
  """The set of screen feature layers."""
  pass

class MinimapFeatures(collections.namedtuple("MinimapFeatures", [
    "height_map", "visibility_map", "creep", "camera", "player_id",
    "player_relative", "selected"])):
  """The set of minimap feature layers."""
  pass

SCREEN_FEATURES = ScreenFeatures(
    height_map=(256, FeatureType.SCALAR),
    visibility_map=(4, FeatureType.CATEGORICAL),
    creep=(2, FeatureType.CATEGORICAL),
    power=(2, FeatureType.CATEGORICAL),
    player_id=(17, FeatureType.CATEGORICAL),
    player_relative=(5, FeatureType.CATEGORICAL),
    unit_type=(1850, FeatureType.CATEGORICAL),
    selected=(2, FeatureType.CATEGORICAL),
    unit_hit_points=(1600, FeatureType.SCALAR),
    unit_energy=(1000, FeatureType.SCALAR),
    unit_shields=(1000, FeatureType.SCALAR),
    unit_density=(16, FeatureType.SCALAR),
    unit_density_aa=(256, FeatureType.SCALAR),
)

MINIMAP_FEATURES = MinimapFeatures(
    height_map=(256, FeatureType.SCALAR),
    visibility_map=(4, FeatureType.CATEGORICAL),
    creep=(2, FeatureType.CATEGORICAL),
    camera=(2, FeatureType.CATEGORICAL),
    player_id=(17, FeatureType.CATEGORICAL),
    player_relative=(5, FeatureType.CATEGORICAL),
    selected=(2, FeatureType.CATEGORICAL),
)

class TimeStep(collections.namedtuple(
    'TimeStep', ['step_type', 'reward', 'discount', 'observation'])):
  """Returned with every call to `step` and `reset` on an environment.

  A `TimeStep` contains the data emitted by an environment at each step of
  interaction. A `TimeStep` holds a `step_type`, an `observation`, and an
  associated `reward` and `discount`.

  The first `TimeStep` in a sequence will have `StepType.FIRST`. The final
  `TimeStep` will have `StepType.LAST`. All other `TimeStep`s in a sequence will
  have `StepType.MID.

  Attributes:
    step_type: A `StepType` enum value.
    reward: A scalar, or `None` if `step_type` is `StepType.FIRST`, i.e. at the
      start of a sequence.
    discount: A discount value in the range `[0, 1]`, or `None` if `step_type`
      is `StepType.FIRST`, i.e. at the start of a sequence.
    observation: A NumPy array, or a dict, list or tuple of arrays.
  """
  __slots__ = ()

  def first(self):
    return self.step_type is StepType.FIRST

  def mid(self):
    return self.step_type is StepType.MID

  def last(self):
    return self.step_type is StepType.LAST

class Wrapper:
    """
    Description:
        This class is used to generate images utilized by the neural network
    Atributes:
        defender_count : int
            Contains the amount of defenders in the simulation
        target_count : int
            Contains the amount of attackers in the simulation
        image_size : int
            Determines the size of the generated images (image_size x image_size)
        maxCoordinate : int
            The simulation has a max size of 1000 for its coordinate system, used to determine ratio.
        ratio : float
            Used to multiply given feature coordinates to scale them to the desired image size.

        defender_detection_image : numpy.array(image_size, image_size) int
            Categorical image (1 or 0) containing the detection radius of all the defenders.
        self.defender_interception_image : numpy.array(image_size, image_size) int
            Categorical image containing the interception radius of all defenders, 1 showing the radius and 0 showing where it is not.
        self.defender_position_image : numpy.array(image_size, image_size) int
            Categorical image containing the current position of each defender, 1 where a defender is.
        self.target_position_image : numpy.array(image_size, image_size) int
            Categorical image containing the current position of each target
        self.target_value_image : numpy.array(image_size, image_size) int
            Scalar image containing the value of each target at their corresponding location.
        self.attacker_position_image : numpy.array(3, image_size, image_size) int
            Categorical image containing the location of each attacker(missile) and holds the previous 3 steps to determine direction of missile
        self.cml_position_image : numpy.array(image_size, image_size) int
            Categorical image containing the location of the cruise missile launcher.
        self.opportunity_position_image : numpy.array(image_size, image_size) int
            Categorical image the contains the position of each chosen opportunity. NOTE - (CHOSEN OR NOT CHOSEN??) Grabbed from defender_opportunties attribute in simulation
        self.opportunity_choice_image : numpy.array(image_size, image_size) int
            Categorical image the contains the position of all possible opportunities. NOTE - Grabbed from defender_actions in Steps method.
    """

    def __init__(self, target_count, defender_count, discount, image_size=64):
        """
        input parameters:
            target_count : int
                Number of targets
            defender_count : int
                Number of defenders
            image_size : int
                Size of the generated images
        """
        self.state = StepType.FIRST
        self.discount = discount
        self.defender_count = defender_count
        self.target_count = target_count
        self.image_size = image_size
        self.maxCoordinate = 1000
        self.ratio = image_size/self.maxCoordinate

        self.defender_detection_image = numpy.zeros((self.image_size, self.image_size), dtype=int)
        self.defender_interception_image = numpy.zeros((self.image_size, self.image_size), dtype=int)
        self.defender_position_image = numpy.zeros((self.image_size, self.image_size), dtype=int)
        self.target_position_image = numpy.zeros((self.image_size, self.image_size), dtype=int)
        self.attacker_position_image = numpy.zeros((3, self.image_size, self.image_size), dtype=int)
        self.cml_position_image = numpy.zeros((self.image_size, self.image_size), dtype=int)
        self.opportunity_position_image = numpy.zeros((self.image_size, self.image_size), dtype=int)
        self.target_value_image = numpy.zeros((self.image_size, self.image_size), dtype=int)

    def init_input(self, features, detectionRadius=400 , interceptionRadius=160):
        """
             Initializes the Images which dont change after a step in the simulation
         :param features: contains all the necessary features of the simulation
         :param detectionradius: the radius in which a defender can detect. 400km ASSUMPTION FROM SIMULATOR
         :param interceptionRadius: the radius in which a defender can intercept a missile. 160km ASSUMPTION FROM SIMULATOR
         :return: Nothing
         """
        self.update_def_det_image(features, detectionRadius)
        self.update_def_int_image(features, interceptionRadius)
        self.update_def_pos_image(features)
        self.update_target_pos_image(features)
        self.update_cml_pos_image(features)
        self.update_attacker_pos_image(features)
        self.generate_network_input()
        self.write_target_screen()

    #TODO - RECONFIRM THAT DEFENDER OPPORTUNITIES IS THE CORRECT INFORMATION WANTED.
    def step_update(self, features, defender_opportunities):
        """
             updates the necessary images after each step in the simulation
         :param features: contains all the necessary features of the simulation
         :param defender_opportunities: contains all the opportunities a defender can take.
         :return: Nothing
         """
        self.update_def_op_image(defender_opportunities)
        self.update_attacker_pos_image(features)
        self.update_target_value_image(features)
        self.generate_network_input()
        self.write_target_screen()

    def generate_network_input(self):
        """
             Converts the obtained information from the simulation into a new format thats readable
             by the StarCraft 2 environment.
         :return: Nothing
         """
        #Stack collected information from the simulator
        screen = numpy.stack((self.defender_detection_image, self.defender_interception_image,
                                   self.defender_position_image, self.target_position_image,
                                   self.cml_position_image, self.opportunity_position_image,
                                   self.target_value_image), axis=0)

        screen = numpy.concatenate((screen, self.attacker_position_image), axis=0)
        filler = numpy.zeros((13, self.image_size, self.image_size), dtype=int)
        screen = numpy.concatenate((screen, filler), axis=0)
        screen = numpy.expand_dims(screen, axis=0)

        minimap = numpy.zeros((1, 17, self.image_size, self.image_size), dtype=int)

        info = numpy.zeros((1, 524), dtype=int)

        return screen, minimap, info

    def observation_spec(self):
        """The observation spec for the SC2 environment.

        Returns:
          The dict of observation names to their tensor shapes. Shapes with a 0 can
          vary in length, for example the number of valid actions depends on which
          units you have selected.
        """
        screen, minimap, info = self.generate_network_input()

        return {
            "single_select": np.zeros((0, 7), dtype=int),  # Actually only (n, 7) for n in (0, 1)
            "multi_select": np.zeros((0, 7), dtype=int),
            "build_queue": np.zeros((0, 7), dtype=int),
            "cargo": np.zeros((0, 7), dtype=int),
            "cargo_slots_available": np.zeros((1,), dtype=int),
            "screen": screen,
            "minimap": minimap,
            "game_loop": np.zeros((1,), dtype=int),
            "score_cumulative": np.zeros((13,), dtype=int),
            "player": np.zeros((11,), dtype=int),
            "control_groups": np.zeros((10, 2), dtype=int),
            # Need to fill the available_actions with at least one action. (0 = no_op assuming it means no option/action)
            "available_actions": np.zeros((1,), dtype=int),  # original = np.zeros((0,), dtype=int)
        }

    def generate_timestep(self, state, reward, discount):
        """Generates an empty timestep for testing purposes.

        Returns:
          A `TimeStep` namedtuple containing:
            step_type: A `StepType` of `FIRST/MID/LAST`.
            reward: `None`, indicating the reward is undefined.
            discount: `None`, indicating the discount is undefined.
            observation: A NumPy array, or a dict, list or tuple of arrays
              corresponding to `observation_spec()`
        """
        observation = self.observation_spec()
        return (TimeStep(step_type=state, reward=reward, discount=discount, observation=observation),)

    def reset(self, env):
        """Starts a new sequence and returns the first `TimeStep` of this sequence.

        Returns:
          A `TimeStep` namedtuple containing:
            step_type: A `StepType` of `FIRST`.
            reward: `None`, indicating the reward is undefined.
            discount: `None`, indicating the discount is undefined.
            observation: A NumPy array, or a dict, list or tuple of arrays
              corresponding to `observation_spec()`.
        """
        features, defDetectionRadius, defInterceptRadius, reward = env.reset()
        #initialize the new environment.
        self.init_input(features, defDetectionRadius, defDetectionRadius)
        self.state = StepType.FIRST

        return(self.generate_timestep(self.state, reward, self.discount))

    def step(self, env, defender_action, attacker_action):
        """Updates the environment according to the action and returns a `TimeStep`.

        Returns:
          A `TimeStep` namedtuple containing:
            step_type: A `StepType` value.
            reward: Reward at this timestep.
            discount: A discount in the range [0, 1].
            observation: A NumPy array, or a dict, list or tuple of arrays
              corresponding to `observation_spec()`.
        """
        features, defender_opportunities, reward, done, info = env.step(defender_action, attacker_action)
        #Update wrapper images.
        self.step_update(features, defender_opportunities)
        if done:
            self.state = StepType.LAST
        else:
            self.state = StepType.MID

        return(self.generate_timestep(self.state, reward, self.discount))


    def update_def_det_image(self, features, radius):
        """
            Updates the defender detection image.
        :param features: contains the location of each defender
        :param radius: the radius in which a defender can detect
        :return: Nothing
        """
        # xgrid and ygrid are tables containing the x and y coordinates as values
        # mgrid is a mesh creation helper
        xgrid, ygrid = numpy.mgrid[:self.image_size, :self.image_size]
        # change the radius to match wanted image size
        radius = radius * self.ratio

        for idx, defender in enumerate(features.defenders):
            #Update positions to match the image size and flip the y axis
            x, y = defender[0:2] * self.ratio
            y = self.image_size - y
            # circles contains the squared distance to the (x, y) point
            circle = (xgrid - y) ** 2 + (ygrid - x) ** 2
            # circle contains 1's and 0's organized in a circle shape
            # Any value less than radius squared is th circle
            image = circle < radius**2
            # Add all the defenders to one image
            self.defender_detection_image = numpy.add(self.defender_detection_image, image)
            # make all values 1
            self.defender_detection_image = self.defender_detection_image > 0
            # change circle data type from true/false to 0/1
            self.defender_detection_image = self.defender_detection_image.astype(int)

    def update_def_op_image(self, defender_opportunities):
        """
            Updates the opportunity position image.
        :param defender_opportunities: contains the location of each opportunity
        :return: Nothing
        """
        self.opportunity_position_image = numpy.zeros((self.image_size, self.image_size), dtype=int)

        for idx, defender in enumerate(defender_opportunities):
            for idy, attacker in enumerate(defender_opportunities[idx]):
                if attacker[0] == 1:
                    x, y = attacker[2:4] * self.ratio
                    x = int(x)
                    y = int(y)
                    y = self.image_size - y

                    if y == self.image_size:
                        y =- 1

                    self.opportunity_position_image[y][x] = 1

                if attacker[4] == 1:
                    x, y = attacker[6:8] * self.ratio
                    x = int(x)
                    y = int(y)
                    y = self.image_size - y

                    if y == self.image_size:
                        y = - 1

                    self.opportunity_position_image[y][x] = 1


    def update_def_int_image(self, features, radius):
        """
            Updates the defender interception image.
        :param features: contains the location of the defenders
        :param radius: the radius in which a defender can detect
        :return: Nothing
        """

        # xgrid and ygrid are tables containing the x and y coordinates as values
        # mgrid is a mesh creation helper
        xgrid, ygrid = numpy.mgrid[:self.image_size, :self.image_size]
        # change the radius to match wanted image size
        radius = radius * self.ratio

        for idx, defender in enumerate(features.defenders):
            x, y = defender[0:2] * self.ratio
            y = self.image_size - y
            # circles contains the squared distance to the (x, y) point
            circle = (xgrid - y) ** 2 + (ygrid - x) ** 2
            # circle contains 1's and 0's organized in a circle shape
            # Any value less than radius squared is th circle
            image = circle < radius ** 2
            # Add all the defenders to one image
            self.defender_interception_image = numpy.add(self.defender_interception_image, image)
            # make all values 1
            self.defender_interception_image = self.defender_interception_image > 0
            # change circle data type from true/false to 0/1
            self.defender_interception_image = self.defender_interception_image.astype(int)

    def update_def_pos_image(self, features):
        """
            Updates the defender interception image.
        :param features: contains the location of the defenders
        :param radius: the radius in which a defender can detect
        :return: Nothing
        """

        #Reset current image back to zeros
        self.defender_position_image = numpy.zeros((self.image_size, self.image_size), dtype=int)

        #loop through each defender grabing their coordinates updating them to match desired image size.
        for defender in features.defenders:
            x, y = defender[0:2] * self.ratio
            x = int(x)
            y = int(y)
            #flip y axis to match simulation
            y = self.image_size - y

            #If coordinate goes over array size (fixable) easy
            if y == self.image_size:
                y =- 1

            self.defender_position_image[y][x] = 1

    def update_target_pos_image(self, features):
        """
            Updates the target position image.
        :param features: contains all the necessary features of the simulation
        :return: Nothing
        """

        self.target_position_image = numpy.zeros((self.image_size, self.image_size), dtype=int)

        for target in features.targets:
            x, y = target[0:2] * self.ratio
            x = int(x)
            y = int(y)
            y = self.image_size - y

            if y == self.image_size:
                y =- 1

            self.target_position_image[y][x] = 1

    def update_attacker_pos_image(self, features):
        """
            Updates the attacker position image. While containing the previous 2 timesteps in order
            to determine the direction in which the missile is moving
        :param features: contains all the necessary features of the simualtion
        :return: Nothing
        """

        self.attacker_position_image[2] = self.attacker_position_image[1]
        self.attacker_position_image[1] = self.attacker_position_image[0]
        self.attacker_position_image[0] = numpy.zeros((self.image_size, self.image_size), dtype=int)

        for attacker in features.attackers:
            x, y = attacker[0:2] * self.ratio
            x = int(x)
            y = int(y)
            y = self.image_size - y

            if y == self.image_size:
                y =- 1

            if attacker[6] == 1:
                self.attacker_position_image[0][y][x] = 1

    def update_cml_pos_image(self, features):
        """
            Updates the cruise missile launcher position image.
        :param features: contains all the necessary features of the simualtion
        :return: Nothing
        """
        self.cml_position_image = numpy.zeros((self.image_size, self.image_size), dtype=int)

        for cml in features.cml:
            x, y = cml[0:2] * self.ratio
            x = int(x)
            y = int(y)
            y = self.image_size - y

            if y == self.image_size:
                y =- 1

            self.cml_position_image[y][x] = 1

    def update_target_value_image(self, features):
        """
            Updates the target value image.
        :param features: contains all the necessary features of the simualtion
        :return: Nothing
        """
        self.target_value_image = numpy.zeros((self.image_size, self.image_size), dtype=int)

        for target in features.targets:

            x, y = target[0:2] * self.ratio
            x = int(x)
            y = int(y)
            y = self.image_size - y

            if y == self.image_size:
                y = - 1

            self.target_value_image[y][x] = int(target[2])

    def write_defender_detection(self):
        """
            Used to write a image to a txt document for testing purposes.
            (easier manipulation and viewing)
        """

        f = open("./wrapper_images/defender_detection_image.txt", "w")
        for x in self.defender_detection_image:
            f.write(str(x))
        f.close()

        f = open("./wrapper_images/defender_detection_image.txt", "r")
        content = f.read()
        f.close()

        f = open("./wrapper_images/defender_detection_image.txt", "w")
        for x in content:
            if x != "\n" and x != " ":
                f.write(x)
            if x == "]":
                f.write("\n")
        f.close()

    def write_defender_interception(self):
        """
            Used to write a image to a txt document for testing purposes.
            (easier manipulation and viewing)
        """

        f = open("./wrapper_images/defender_interception_image.txt", "w")
        for x in self.defender_interception_image:
            f.write(str(x))
        f.close()

        f = open("./wrapper_images/defender_interception_image.txt", "r")
        content = f.read()
        f.close()

        f = open("./wrapper_images/defender_interception_image.txt", "w")
        for x in content:
            if x != "\n" and x != " ":
                f.write(x)
            if x == "]":
                f.write("\n")
        f.close()

    def write_defender_postions(self):
        """
            Used to write a image to a txt document for testing purposes.
            (easier manipulation and viewing)
        """

        f = open("./wrapper_images/defender_position_image.txt", "w")
        for x in self.defender_position_image:
            f.write(str(x))
        f.close()

        f = open("./wrapper_images/defender_position_image.txt", "r")
        content = f.read()
        f.close()

        f = open("./wrapper_images/defender_position_image.txt", "w")
        for x in content:
            if x != "\n" and x != " ":
                f.write(x)
            if x == "]":
                f.write("\n")
        f.close()

    def write_target_postions(self):
        """
            Used to write a image to a txt document for testing purposes.
            (easier manipulation and viewing)
        """

        f = open("./wrapper_images/target_position_image.txt", "w")
        for x in self.target_position_image:
            f.write(str(x))
        f.close()

        f = open("./wrapper_images/target_position_image.txt", "r")
        content = f.read()
        f.close()

        f = open("./wrapper_images/target_position_image.txt", "w")
        for x in content:
            if x != "\n" and x != " ":
                f.write(x)
            if x == "]":
                f.write("\n")
        f.close()

    def write_attacker_postions(self):
        """
            Used to write a image to a txt document for testing purposes.
            (easier manipulation and viewing)
        """

        f = open("./wrapper_images/attacker_position_image.txt", "w")
        for x in self.attacker_position_image:
            f.write(str(x))
        f.close()

        f = open("./wrapper_images/attacker_position_image.txt", "r")
        content = f.read()
        f.close()

        f = open("./wrapper_images/attacker_position_image.txt", "w")
        for x in content:
            if x != "\n" and x != " ":
                f.write(x)
            if x == "]":
                f.write("\n")
        f.close()

    def write_cml_postions(self):
        """
            Used to write a image to a txt document for testing purposes.
            (easier manipulation and viewing)
        """

        f = open("./wrapper_images/cml_position_image.txt", "w")
        for x in self.cml_position_image:
            f.write(str(x))
        f.close()

        f = open("./wrapper_images/cml_position_image.txt", "r")
        content = f.read()
        f.close()

        f = open("./wrapper_images/cml_position_image.txt", "w")
        for x in content:
            if x != "\n" and x != " ":
                f.write(x)
            if x == "]":
                f.write("\n")
        f.close()

    def write_def_opportunities(self):
        """
            Used to write a image to a txt document for testing purposes.
            (easier manipulation and viewing)
        """

        f = open("./wrapper_images/def_opportunities_image.txt", "w")
        for x in self.opportunity_position_image:
            f.write(str(x))
        f.close()

        f = open("./wrapper_images/def_opportunities_image.txt", "r")
        content = f.read()
        f.close()

        f = open("./wrapper_images/def_opportunities_image.txt", "w")
        for x in content:
            if x != "\n" and x != " ":
                f.write(x)
            if x == "]":
                f.write("\n")
        f.close()

    def write_target_value(self):
        """
            Used to write a image to a txt document for testing purposes.
            (easier manipulation and viewing)
        """

        f = open("./wrapper_images/target_value_image.txt", "w")
        for x in self.target_value_image:
            f.write(str(x))
        f.close()

        f = open("./wrapper_images/target_value_image.txt", "r")
        content = f.read()
        f.close()

        f = open("./wrapper_images/target_value_image.txt", "w")
        for x in content:
            if x != "\n" and x != " ":
                f.write(x)
            if x == "]":
                f.write("\n")
        f.close()

    def write_target_screen(self):
        """
            Used to write a image to a txt document for testing purposes.
            (easier manipulation and viewing)
        """
        screen, mini, info = self.generate_network_input()
        f = open("./wrapper_images/screen.txt", "w")
        for x in screen:
            f.write(str(x))
        f.close()

        f = open("./wrapper_images/screen.txt", "r")
        content = f.read()
        f.close()

        f = open("./wrapper_images/screen.txt", "w")
        for x in content:
            if x != "\n" and x != " ":
                f.write(x)
            if x == "]":
                f.write("\n")
        f.close()