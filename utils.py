from math import sqrt
import numpy as np


def euclidean_distance(position1, position2):
    """
    input parameters:
        position1: x, y coordinates
        position2: x, y coordinates
    return:
        The euclidean distance between the two positions
    """
    return sqrt((float(position1[0])-float(position2[0]))**2 + (float(position1[1]) - float(position2[1]))**2)

def pos_to_dist(x_vals, y_vals):
    """
    Same as euclidean distance with a 0,0 position, but works with numpy columns of x, y values.
    input parameters:
        x_vals : x coordinate(s)
        y_vals : y coordinate(s)
    return:
        distance(s) from origin to point(s)
    """
    dist = (x_vals**2 + y_vals**2)**0.5
    return dist


def detection_pairs(interceptor_positions, missile_positions, detection_radius):
    """
    input parameters:
        interceptor_positions: The x, y coordinates of interceptors
        missile_positions: The x, y coordinates of missiles
        detection_radius: The radius within which an interceptor can detect a missile.
    return:
        An array containing indeces which indicate the interceptor row numbers and corresponding detected missile rows
        in column 0 and 1 respectively.
    """
    distance = np.sqrt(np.sum(np.power(np.abs(interceptor_positions[:, None] - missile_positions[None, :]), 2), axis=2))
    return np.column_stack((np.where(distance < detection_radius)[0], np.where(distance < detection_radius)[1]))


def decomposed_velocity(velocity, origin, destination, time_step_size):
    """
    Decomposes velocity into x, y components and calculates how many time steps it will take to reach destination.
    input parameters:
        velocity : numeric
            The velocity to be decomposed (km/min)
        origin : array-like
            The origin of the object (x, y coordinates in km)
        destination : array-like
            The destination of the object (x, y coordinates in km)
        time_step_size :  numeric
            The size of a time step in minutes
    returns:
        delta_x : numeric
            The distance travelled in km per time step, in the x direction.
        delta_y : numeric
            The distance travelled in km per time step, in the y direction.
        time_steps_to_hit : numeric
            The number of time steps until object reaches destination
    """
    distance_per_ts = velocity*time_step_size
    ptp_dist = euclidean_distance(origin, destination)
    percent_dist = distance_per_ts/ptp_dist
    time_steps_to_hit = abs(ptp_dist/distance_per_ts)
    delta_x = percent_dist*(destination[0] - origin[0])
    delta_y = percent_dist*(destination[1] - origin[1])
    return delta_x, delta_y, time_steps_to_hit


def sort(events):
    """
    An insertion sort on the events list.
    input parameters:
        events: A list of lists that we need sorted by the zeroth index of the sub-lists
    """
    for i in range(1, len(events)):
        key = events[i]
        j = i - 1
        while j >= 0 and key[0] < events[j][0]:
            events[j + 1] = events[j]
            j -= 1
        events[j + 1] = key