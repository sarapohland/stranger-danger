import warnings
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.interpolate import interp1d


def reached_destination(robot, goal_radius):
    dist_goal = np.linalg.norm(np.array(robot.get_position()) - np.array(robot.get_goal_position()))
    return  dist_goal < robot.radius + goal_radius

def generate_wall(vertices):
    return Polygon(vertices)

def generate_room(map_name):
    half_w = 0.05
    walls = []
    # create walls from the appropriate map file
    with open("../crowd_demo/maps/{}.txt".format(map_name), "r") as map_file:
        print("Making {} map".format(map_name))
        for line in map_file:
            splitty = line.split(',')
            direction = splitty[0]
            coords = [float(splitty[i]) for i in range(1, 5)]

            # horizontal line (assumes coords are left, right)
            if direction == 'h':
                if coords[1] != coords[3]:
                    warnings.warn('this is not a horizontal line lmao')
                walls.append(generate_wall([(coords[0]-half_w, coords[1]-half_w), (coords[0]-half_w, coords[1]+half_w),
                                            (coords[2]+half_w, coords[3]+half_w), (coords[2]+half_w, coords[3]-half_w)]))
            # vertical line (assumes coords are bottom, top)
            elif direction == 'v':
                if coords[0] != coords[2]:
                    warnings.warn('this is not a vertical line lmao')
                walls.append(generate_wall([(coords[0]+half_w, coords[1]-half_w), (coords[0]-half_w, coords[1]-half_w),
                                            (coords[2]-half_w, coords[3]+half_w), (coords[2]+half_w, coords[3]+half_w)]))
            # block (assumes coords are lower left, upper right)
            elif direction == 'b':
                walls.append(generate_wall([(coords[0], coords[1]), (coords[0], coords[3]), (coords[2], coords[3]), (coords[2], coords[1])]))
            else:
                warnings.warn("the first character in each line of a map file must be 'h' for a horizontal wall, 'v' for a vertical wall, or 'b' for a block")

    # Take the union of all walls to generate one wall Polygon object
    return unary_union(walls)

def get_path_details(path):
    path_counter = 1
    with open("../hardware/maps/demo_path.txt", "r") as path_file:
        for line in path_file:
            if path_counter == path:
                splitty = line.split(',')
                return splitty[0], float(splitty[1]), float(splitty[2]), float(splitty[3]), float(splitty[4])
            path_counter += 1

    warnings.warn('this path does not exist')
    return None, 0, 0, 0, 0

def load_waypoints(path, map_name, config, end):
    waypoint_radius = config.getfloat('waypoints', 'waypoint_radius')
    waypoint_min_spacing = config.getfloat('waypoints', 'waypoint_min_spacing')
    waypoint_max_spacing = config.getfloat('waypoints', 'waypoint_max_spacing')
    vary_spacing = config.getboolean('waypoints', 'vary_spacing')
    vary_degrees = config.getfloat('waypoints', 'vary_degrees')

    # load in waypoints from the desired file
    waypoints = []
    with open("../hardware/waypoints/{}{}.txt".format(map_name, path), 'r') as waypoint_file:
        for line in waypoint_file:
            waypoints.append([float(coord) for coord in line.split(',')])
    # for some reason the waypoints generation code saved all the waypoints in reverse order T-T
    waypoints.reverse()
    waypoints = np.array(waypoints)

    # conduct a spline interpolation to construct waypoints that are equally spaced apart
    t = [0]
    for i in range(1, len(waypoints)):
        t.append(t[i-1] + np.linalg.norm(waypoints[i] - waypoints[i-1]))
    t = np.array(t)

    # generate nicely-spaced waypoints
    if vary_spacing:
        num_new_wp = int(t[-1] / waypoint_min_spacing) + 1
    else:
        num_new_wp = int(t[-1] / waypoint_min_spacing) + 1
    t_new = np.linspace(t[0], t[-1], num=num_new_wp, endpoint=True)

    x = waypoints[:,0]
    y = waypoints[:,1]

    fx = interp1d(t, x, kind='quadratic') #'cubic')
    fy = interp1d(t, y, kind='quadratic') #'cubic')

    if vary_spacing:
        x_new = fx(t_new)
        y_new = fy(t_new)
        max_rad = vary_degrees * np.pi / 180.0
        last_wypt = np.array([x_new[0], y_new[0]])
        cur_vec = np.array([x_new[1], y_new[1]]) - last_wypt
        cur_dir = cur_vec / np.linalg.norm(cur_vec)
        result = [[x_new[0], y_new[0]]]
        for i in range(2, num_new_wp):
            next_vec = np.array([x_new[i], y_new[i]]) - last_wypt
            next_dist = np.linalg.norm(next_vec)
            next_dir = next_vec / next_dist
            # If this waypoint is too far from the current direction, append the previous waypoint, which should be
            # within the acceptable deviation from the current direction.
            if np.arccos(np.min([1, np.dot(cur_dir, next_dir)])) > max_rad or next_dist > waypoint_max_spacing:
                result.append([x_new[i-1], y_new[i-1]])
                last_wypt = np.array([x_new[i-1], y_new[i-1]])
                cur_vec = np.array([x_new[i], y_new[i]]) - last_wypt
                cur_dir = cur_vec / np.linalg.norm(cur_vec)
        result.append([x_new[-1], y_new[-1]])
        return result
    else:
        return [[new_x, new_y] for new_x, new_y in zip(fx(t_new), fy(t_new))]
