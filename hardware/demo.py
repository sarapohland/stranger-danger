import argparse
import configparser
import os
import time
import torch
import imageio
import numpy as np
import copy

from shapely import affinity
from shapely.geometry import Polygon
from matplotlib import patches
from matplotlib import pyplot as plt
from descartes.patch import PolygonPatch
from skimage.measure import block_reduce
from dotmap import DotMap
from pytorchyolo import detect, models

from control.policy.policy_factory import policy_factory
from simulation.envs.utils.robot import Robot
from simulation.envs.utils.human import Human
from hardware.utils.turtlebot_hardware import TurtlebotHardware
from hardware.utils.utils import *
from uncertainty.estimate_epsilons import estimate_epsilons


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='../crowd_nav/configs/env.config')
    parser.add_argument('--policy_config', type=str, default='../crowd_nav/configs/policy.config')
    parser.add_argument('--policy', type=str, default='uncertain_sarl')
    parser.add_argument('--model_dir', type=str, default='../crowd_nav/model/uncertain')
    parser.add_argument('--phase', type=str, default='demo')
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.policy_config

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights))

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    policy.set_phase(args.phase)
    policy.set_device(device)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    time_step = env_config.getfloat('env', 'time_step')
    path_number = env_config.getint('waypoints', 'path')
    wypt_radius = env_config.getfloat('waypoints', 'waypoint_radius')
    goal_radius = env_config.getfloat('reward', 'goal_radius')

    # configure robot
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    robot.time_step = time_step
    robot.policy.time_step = time_step

    # initialize robot
    map_name, px, py, gx, gy = get_path_details(path_number)
    walls = generate_room(map_name)
    robot.policy.set_walls(walls)
    robot_origin = np.array([px, py])
    robot.set(px, py, gx, gy, 0, 0, 0)
    robot.set_waypoints(load_waypoints(path_number, map_name, env_config, [gx, gy]), wypt_radius)

    # set up object detection
    yolo = models.load_model('yolo/yolov3.cfg', 'yolo/yolov3.weights')
    obstacle_ids = [0] # human=0, chair=56, backpack=24

    # initilaize turtlebot hardware
    params = DotMap(dt=time_step)
    turtlebot = TurtlebotHardware(params)
    average_time_step = 0.5

    # intialize list of objects and humans
    objects = []
    humans = []
    object_detect_times = []
    human_detect_times = []
    start_time = time.time()
    last_time = start_time

    # initialize the camera feed if displaying sim
    if args.display:
        fig, (rgb_ax, depth_ax, sim_ax) = plt.subplots(1, 3, figsize=(26,8))
        sim_ax.set_aspect('equal')
        plt.ion()
        plt.show()

    # initialize image frames if saving video
    if args.save_video:
        rgb_frames = []
        depth_frames = []
        object_boxes_frames = []
        human_boxes_frames = []
        object_states_frames = []
        human_states_frames = []
        robot_states_frames = []

    # save human positions over time
    human_positions = []

    # obstacle tracking parameters
    detect_range = 8 # meters
    sarl_range = 4 # meters
    object_keep_time = 4 # seconds per time seen
    human_keep_time = 2 # seconds per time seen
    first_sight_time = 1 # seconds

    # camera calibration parameters
    fov = 59.30356 * np.pi / 180 # radians
    offset = 0.381 # meters
    dist_scale = 1.06
    angle_scale = 0.05 / (20 * np.pi / 180)

    # while the robot has not reached its goal or hit an obstacle...
    while not reached_destination(robot, goal_radius) and not turtlebot.hit_obstacle:
        # detect obstacles
        boxes, obs_attributes = [], []
        rgb = np.asarray(turtlebot.rgb_raw_image)
        depth = np.asarray(turtlebot.depth_raw_image)

        # determine robot position, velocity, and angle
        robot.set_position(turtlebot.position + robot_origin)
        robot.set_velocity(turtlebot.velocity)
        robot.set_angle(turtlebot.angle)

        if rgb is not None and depth is not None:
            # downsample the images for faster yolo
            #rgb = block_reduce(rgb, block_size=(2,2,1), func=block_func).astype(np.uint8)
            #depth = block_reduce(depth, block_size=(2,2), func=block_func)

            # display image if necessary
            if args.display:
                rgb_ax.clear()
                rgb_ax.imshow(rgb)
                depth_ax.clear()
                depth_ax.imshow(depth)

            # save the image frames and instantiate lists for humans and objects in this frame
            if args.save_video:
                rgb_frames.append(rgb)
                depth_frames.append(depth)
                human_boxes_frame = []
                object_boxes_frame = []

            # use yolo to detect obstacles in image
            IMG_WIDTH, IMG_HEIGHT = rgb.shape[1], rgb.shape[0]
            boxes = detect.detect_image(yolo, rgb)

            # create a distance matrix for existing humans (if any)
            human_dists = None
            if len(humans) > 0 and len(boxes) > 0:
                human_dists = np.full((len(boxes), len(humans)), np.inf)

            # create a distance matrix for existing objects (if any)
            object_dists = None
            if len(objects) > 0 and len(boxes) > 0:
                object_dists = np.full((len(boxes), len(objects)), np.inf)

            # go through each obstacle detected by yolo...
            for i, box in enumerate(boxes):
                # ignore detected obstacles we don't care about
                if int(box[5]) not in obstacle_ids:
                    obs_attributes.append(None)
                    continue

                # get corners of bounding box for obstacle
                x0 = np.maximum(int(box[0]), 0)
                y0 = np.maximum(int(box[1]), 0)
                x1 = np.minimum(int(box[2]), IMG_WIDTH-1)
                y1 = np.minimum(int(box[3]), IMG_HEIGHT-1)

                # get angle of obstacle wrt robot
                center = ((x1 - x0)/2 + x0)
                angle = (center / IMG_WIDTH * fov - fov/2)

                # get distance of obstacle from robot
                region = copy.copy(depth[y0:y1, x0:x1])
                # check if obstacle is obstructed by another obstacle
                box_poly = Polygon([(x0,y0), (x0,y1), (x1,y1), (x1,y0)])
                for j, other_box in enumerate(boxes):
                    if i != j:
                        other_x0 = np.maximum(int(other_box[0]), 0)
                        other_y0 = np.maximum(int(other_box[1]), 0)
                        other_x1 = np.minimum(int(other_box[2]), IMG_WIDTH-1)
                        other_y1 = np.minimum(int(other_box[3]), IMG_HEIGHT-1)
                        other_box_poly = Polygon([(other_x0,other_y0), (other_x0,other_y1), (other_x1,other_y1), (other_x1,other_y0)])
                        if box_poly.intersects(other_box_poly) and y0 > other_y0:
                            region[y0:other_y1, x0:other_x1] = 0
                nonzeros = (region != 0)

                # if we don't have a good distance estimate for obstacle, ignore it
                if np.sum(nonzeros) <= 200:
                    obs_attributes.append(None)
                    continue
                dist = np.median(region[nonzeros]) / 1000
                dist = (dist_scale + (angle_scale * np.abs(angle))) * dist

                # if this obstacle is too far away, ignore it
                if dist > detect_range:
                    obs_attributes.append(None)
                    continue

                # get position of obstacle in robot frame
                theta = angle + np.arcsin(offset * np.sin(angle) / dist)
                x = dist * np.cos(theta) # robot frame
                y = -dist * np.sin(theta) # robot frame

                # get position of obstacle in world frame
                px = x * np.cos(robot.theta) - y * np.sin(robot.theta) + robot.px # world frame
                py = x * np.sin(robot.theta) + y * np.cos(robot.theta) + robot.py # world frame

                # get radius of obstacle
                dx = x1 - x0
                width = 2 * (x + offset) * np.tan(fov/2)
                radius = (dx / IMG_WIDTH) * width / 2

                # create list of attributes for each obstacle (position, radius)
                obs_attributes.append([px, py, radius])

                # if we detected an object...
                if int(box[5]) != 0:
                    # outline the object in red
                    if args.display:
                        rgb_ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor='r', facecolor='none'))
                        depth_ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor='r', facecolor='none'))
                    if args.save_video:
                        object_boxes_frame.append([(x0, y0), x1-x0, y1-y0])

                    # check if the object intersects with an existing object
                    for j, object in enumerate(objects):
                        # if the detected object collides with an existing object, add it to our distance matrix
                        cur_dist = np.linalg.norm((px - object.px, py - object.py))
                        if cur_dist < object.radius + radius:
                            object_dists[i, j] = cur_dist

                # if we detected a human...
                else:
                    # outline the human in green
                    if args.display:
                        rgb_ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor='g', facecolor='none'))
                        depth_ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor='g', facecolor='none'))
                    if args.save_video:
                        human_boxes_frame.append([(x0, y0), x1-x0, y1-y0])

                    # check if the human intersects with an existing human
                    for j, human in enumerate(humans):
                        # if the detected human collides with an existing human, add it to our distance matrix
                        cur_dist = np.linalg.norm((px - human.px, py - human.py))
                        if cur_dist < human.radius + radius + 1*(np.linalg.norm(human.get_velocity()) == 0):
                            human_dists[i, j] = cur_dist

            mapped_humans = []
            # map known humans to closest observed humans using distance matrix
            if human_dists is not None:
                cur_min_dist = np.min(human_dists)
                cur_min_index = np.unravel_index(np.argmin(human_dists), human_dists.shape)
                while cur_min_dist < np.inf:
                    # update the known human with the observed values
                    mapped_humans.append(cur_min_index[0])
                    times_seen = human_detect_times[cur_min_index[1]][1] + 1
                    human_detect_times[cur_min_index[1]] = (time.time() - start_time + times_seen*human_keep_time, times_seen)
                    human = humans[cur_min_index[1]]
                    hum_attr = obs_attributes[cur_min_index[0]]
                    new_vx = human.vx + 0.5 * (hum_attr[0] - human.px)/average_time_step
                    new_vy = human.vy + 0.5 * (hum_attr[1] - human.py)/average_time_step
                    new_radius = 0.8 * human.radius + 0.2 * hum_attr[2]
                    human.set(hum_attr[0], hum_attr[1], 0, 0, new_vx, new_vy, np.arctan2(new_vy, new_vx), radius=new_radius)
                    human_positions[cur_min_index[1]].append(human.get_position())

                    # update our matrix and run it again
                    human_dists[cur_min_index[0], :] = np.inf
                    human_dists[:, cur_min_index[1]] = np.inf
                    cur_min_dist = np.min(human_dists)
                    cur_min_index = np.unravel_index(np.argmin(human_dists), human_dists.shape)

            mapped_objects = []
            # map known objects to closest observed objects using distance matrix
            if object_dists is not None:
                cur_min_dist = np.min(object_dists)
                cur_min_index = np.unravel_index(np.argmin(object_dists), object_dists.shape)
                while cur_min_dist < np.inf:
                    # update the known object with the observed values
                    mapped_objects.append(cur_min_index[0])
                    times_seen = object_detect_times[cur_min_index[1]][1] + 1
                    object_detect_times[cur_min_index[1]] = (time.time() - start_time + times_seen*object_keep_time, times_seen)
                    object = objects[cur_min_index[1]]
                    obj_attr = obs_attributes[cur_min_index[0]]
                    new_px = 0.8 * object.px + 0.2 * obj_attr[0]
                    new_py = 0.8 * object.py + 0.2 * obj_attr[1]
                    new_radius = 0.8 * object.radius + 0.2 * obj_attr[2]
                    object.set(new_px, new_py, 0, 0, 0, 0, 0, radius=new_radius)

                    # update our matrix and run it again
                    object_dists[cur_min_index[0], :] = np.inf
                    object_dists[:, cur_min_index[1]] = np.inf
                    cur_min_dist = np.min(object_dists)
                    cur_min_index = np.unravel_index(np.argmin(object_dists), object_dists.shape)

            for index, box in enumerate(boxes):
                # if we are detected an object of interest and did not map them, instantiate a new object
                if int(box[5]) in obstacle_ids and int(box[5]) != 0 and index not in mapped_objects and obs_attributes[index] is not None:
                    new_object = Object(env_config, 'objects')
                    new_object.set(obs_attributes[index][0], obs_attributes[index][1], 0, 0, 0, 0, 0, radius=obs_attributes[index][2])
                    objects.append(new_object)
                    object_detect_times.append((time.time() - start_time + first_sight_time, 1))

                # if we are detected a human of interest and did not map them, instantiate a new human
                if int(box[5]) == 0 and index not in mapped_humans and obs_attributes[index] is not None:
                    new_human = Human(env_config, 'humans')
                    new_human.set(obs_attributes[index][0], obs_attributes[index][1], 0, 0, 0, 0, 0, radius=obs_attributes[index][2])
                    humans.append(new_human)
                    human_detect_times.append((time.time() - start_time + first_sight_time, 1))
                    human_positions.append([new_human.get_position()])

        # remove objects that have not been seen recently or are too far away
        obsolete_ids = set()
        current_time = time.time() - start_time
        for i, detect_time in enumerate(object_detect_times):
            if current_time - detect_time[0] > 0:
                obsolete_ids.add(i)
        for i, object in enumerate(objects):
            dist = np.linalg.norm([object.px - robot.px, object.py - robot.py])
            if dist > detect_range:
                obsolete_ids.add(i)
        for i in sorted(obsolete_ids, reverse=True):
            del object_detect_times[i]
            del objects[i]

        # increment humans
        for human in humans:
            human.px += human.vx * average_time_step
            human.py += human.vy * average_time_step

        # remove humans that have not been seen recently or are too far away
        obsolete_ids = set()
        current_time = time.time() - start_time
        for i, detect_time in enumerate(human_detect_times):
            if current_time - detect_time[0] > 0:
                obsolete_ids.add(i)
        for i, human in enumerate(humans):
            dist = np.linalg.norm([human.px - robot.px, human.py - robot.py])
            if dist > detect_range:
                obsolete_ids.add(i)
        for i in sorted(obsolete_ids, reverse=True):
            del human_detect_times[i]
            del humans[i]
            del human_positions[i]

        # refresh robot position, velocity, and angle
        robot.set_position(turtlebot.position + robot_origin)
        robot.set_velocity(turtlebot.velocity)
        robot.set_angle(turtlebot.angle)

        # get robot observation
        agents = objects + humans
        ob = []
        ob_human_positions = []
        robox, roboy = robot.get_position()
        for i, agent in enumerate(agents):
            agentx, agenty = agent.get_position()
            if np.linalg.norm((agentx - robox, agenty - roboy)) < sarl_range:
                ob.append(agent.get_observable_state())
                ob_human_positions.append(human_positions[i])

        # choose optimal action
        eps = estimate_epsilons(ob_human_positions, average_time_step)
        action = robot.act(ob, eps, walls)
        if action is None:
            print("Received a None action!")
            continue
        speed = np.linalg.norm(robot.get_velocity())
        average_time_step = average_time_step * 0.8 + (time.time() - last_time) * 0.2
        last_time = time.time()
        ang_vel = action.r / average_time_step #time_step
        lin_vel = action.v / (1 + 2*np.abs(ang_vel)) # slow down a bit when turning a lot
        lin_vel = np.clip(lin_vel, speed-0.1, speed+0.1) # do not accelerate too much
        ang_vel = np.clip(ang_vel, -1.5, 1.5) # do not turn too fast
        turtlebot.apply_command([lin_vel, ang_vel])

        if args.display or args.save_video:
            action_theta = robot.theta + action.r
            action_dx = action.v * np.cos(action_theta)
            action_dy = action.v * np.sin(action_theta)

        # display where the robot thinks it is
        if args.display:
            # draw where the robot thinks it is and what the robot thinks it sees
            sim_ax.clear()
            sim_ax.set_xlim([robot.px - 5, robot.px + 5])
            sim_ax.set_ylim([robot.py - 5, robot.py + 5])
            sim_ax.add_patch(PolygonPatch(walls))
            sim_ax.add_patch(patches.Circle((robot.px, robot.py), robot.radius, color='red' if robot.policy.safety_active else 'blue'))
            sim_ax.add_artist(plt.arrow(robot.px, robot.py, action_dx, action_dy))
            sim_ax.add_patch(patches.Circle((robot.wx, robot.wy), wypt_radius, color='blue', alpha=0.2))
            for object in objects:
                sim_ax.add_patch(patches.Circle((object.px, object.py), object.radius, color='yellow'))
            for human in humans:
                sim_ax.add_patch(patches.Circle((human.px, human.py), human.radius, color='green'))
                sim_ax.add_artist(plt.arrow(human.px, human.py, human.vx, human.vy))

            # render all the plots
            plt.draw()
            plt.pause(0.0001)

        if args.save_video:
            robot_states_frames.append([robot.px, robot.py, robot.wx, robot.wy, action_dx, action_dy, robot.radius, robot.policy.safety_active])
            object_states_frame = []
            for object in objects:
                object_states_frame.append([(-object.py, object.px), object.radius])
            human_states_frame = []
            for human in humans:
                human_states_frame.append([human.px, human.py, human.vx, human.vy, human.radius])

            # store all of the things for later
            object_boxes_frames.append(object_boxes_frame)
            human_boxes_frames.append(human_boxes_frame)
            object_states_frames.append(object_states_frame)
            human_states_frames.append(human_states_frame)

    for slow_lin in np.linspace(np.linalg.norm(robot.get_velocity()), 0, 8):
        turtlebot.apply_command([slow_lin, 0])

    # render the video and save it if we want to
    if args.save_video:
        print("rendering video...")
        # setup folders for frames
        os.mkdir('tmp')
        frame_name = 'tmp/{}.png'
        frame_names = []

        # instantiate figure
        fig, (rgb_ax, sim_ax, depth_ax) = plt.subplots(3, 1, figsize=(10,20))
        sim_ax.set_aspect('equal')

        # rotate walls 90 degrees
        walls = affinity.rotate(walls, 90, origin=(0,0))

        # render each frame
        for cur_frame in range(len(rgb_frames)):
            # display video feeds
            rgb_ax.clear()
            rgb_ax.imshow(rgb_frames[cur_frame])
            depth_ax.clear()
            depth_ax.imshow(depth_frames[cur_frame])

            # draw YOLO boxes
            for object_box in object_boxes_frames[cur_frame]:
                rgb_ax.add_patch(patches.Rectangle(object_box[0], object_box[1], object_box[2], linewidth=2, edgecolor='r', facecolor='none'))
                depth_ax.add_patch(patches.Rectangle(object_box[0], object_box[1], object_box[2], linewidth=2, edgecolor='r', facecolor='none'))

            # draw where the robot thinks it is and what is around it
            sim_ax.clear()

            # robot
            robot_frame = robot_states_frames[cur_frame] # px, py, wx, wy, vx, vy, radius, safety_active
            sim_ax.set_xlim([-robot_frame[1] - 6, -robot_frame[1] + 6])
            sim_ax.set_ylim([robot_frame[0] - 4.5, robot_frame[0] + 4.5])
            sim_ax.add_patch(PolygonPatch(walls, color='black'))
            sim_ax.add_patch(patches.Circle((-robot_frame[1], robot_frame[0]), robot_frame[6], color='red' if robot_frame[7] else 'blue'))
            sim_ax.plot([-robot_frame[1], -robot_frame[1]-robot_frame[5]], [robot_frame[0], robot_frame[0]+robot_frame[4]], color='black')
            #sim_ax.add_artist(plt.arrow(-robot_frame[1], robot_frame[0], -robot_frame[5], robot_frame[4]))
            sim_ax.add_patch(patches.Circle((-robot_frame[3], robot_frame[2]), wypt_radius, color='blue', alpha=0.2))

            # objects
            for object_frame in object_states_frames[cur_frame]: # (-py, px), radius
                sim_ax.add_patch(patches.Circle(object_frame[0], object_frame[1], color='yellow'))

            # humans
            for human_frame in human_states_frames[cur_frame]: # px, py, vx, vy, radius
                sim_ax.add_patch(patches.Circle((-human_frame[1], human_frame[0]), human_frame[4], color='green'))
                sim_ax.plot([-human_frame[1], -human_frame[1]-human_frame[3]], [human_frame[0], human_frame[0]+human_frame[2]], color='black')
                #sim_ax.add_artist(plt.arrow(-human_frame[1], human_frame[0], -human_frame[3], human_frame[2]))

            # remove tick marks
            sim_ax.axes.xaxis.set_visible(False)
            sim_ax.axes.yaxis.set_visible(False)
            rgb_ax.axes.xaxis.set_visible(False)
            rgb_ax.axes.yaxis.set_visible(False)
            depth_ax.axes.xaxis.set_visible(False)
            depth_ax.axes.yaxis.set_visible(False)

            fig.tight_layout()

            # save the frame
            plt.savefig(frame_name.format(cur_frame), bbox_inches='tight')
            frame_names.append(frame_name.format(cur_frame))

        print("Saving video in video.gif...")
        with imageio.get_writer('video.gif', mode='I', fps=1/average_time_step) as writer:
            for filename in frame_names:
                writer.append_data(imageio.imread(filename))

        # remove files
        for filename in frame_names:
            os.remove(filename)
        os.rmdir('tmp')

if __name__ == '__main__':
    main()
