import numpy
import tensorflow as tf
from network import build_fcn

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

    def __init__(self, target_count, defender_count, image_size=64):
        """
        input parameters:
            target_count : int
                Number of targets
            defender_count : int
                Number of defenders
            image_size : int
                Size of the generated images
        """
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

    def init_input(self, features, detectionRadius, interceptionRadius):
        """
             Initializes the Images which dont change after a step in the simulation
         :param features: contains all the necessary features of the simulation
         :param detectionradius: the radius in which a defender can detect
         :param interceptionRadius: the radius in which a defender can intercept a missile
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