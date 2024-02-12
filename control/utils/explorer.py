import os
import math
import time
import copy
import torch
import pickle
import logging
import numpy as np
import pandas as pd

from simulation.envs.utils.info import *


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0

def curvature(p0, p1, p2):
    area = _get_area(p0, p1, p2)
    d0 = _get_dist(p0, p1)
    d1 = _get_dist(p1, p2)
    d2 = _get_dist(p2, p0)
    curv = 4*area/(d0*d1*d2)
    return 0 if math.isnan(curv) else curv

def _get_area(p0, p1, p2):
    return (p1[0] - p0[0])*(p2[1] - p0[1]) - (p1[1] - p0[1])*(p2[0] - p0[0])

def _get_dist(p0, p1):
    return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def collision_blame(h_pos, h_vel, r_pos, r_vel):
    w = np.array(h_pos) - np.array(r_pos)
    w = w / np.linalg.norm(w)
    robot_blame = r_vel[0] * w[0] + r_vel[1] * w[1]
    human_blame = -(h_vel[0] * w[0] + h_vel[1] * w[1])
    return robot_blame - human_blame

def discomfort_dist(num_hum):
    m = (0.1 - 0.45) / (20 - 6)
    b = 0.1 - 20 * m
    y = m * num_hum + b
    return np.clip(y, 0.1, 0.45)


class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None, max_agents=None, stats_file=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.max_agents = max_agents
        self.target_model = None
        self.stats_file = stats_file

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def run_k_episodes(self, k, phase, update_memory=False, episode=None,
                        imitation_learning=False, print_failure=False, max_epsilon=0, estimate_eps=False):
        self.robot.policy.set_phase(phase)

        success_times = []
        timeout_times = []
        collision_times = []

        min_dist = []
        min_dist_danger = []
        cumulative_rewards = []

        timeout_cases = []
        collision_cases = []
        # collision_positions = []

        avg_robot_vel = []
        avg_robot_acc = []
        avg_robot_jer = []
        avg_human_vel = []
        avg_human_acc = []
        avg_human_jer = []

        success = 0
        collision = 0
        timeout = 0
        too_close = 0

        stats = {'test_case':[],'scenario':[],'perpetual':[],'num_humans':[],'open_space':[],'navigation_time':[],
                      'result':[],'num_collisions':[],'collision_times':[],'collision_positions':[],'collision_blames':[],
                      'inference_time_mean':[],'inference_time_std':[],'inference_time_min':[],'inference_time_max':[],
                      'min_dist_mean':[],'min_dist_std':[],'min_dist_min':[],
                      'vel_mean':[],'vel_std':[],'acc_mean':[],'acc_std':[],'jerk_mean':[],'jerk_std':[],
                      'curvature_mean': [], 'curvature_std': [],
                      'time_intruded':[],'intruded_min_dist_mean':[],'intruded_min_dist_std':[],
                      'intruded_vel_mean':[],'intruded_vel_std':[],'intruded_acc_mean':[],
                      'intruded_acc_std':[],'intruded_jerk_mean':[],'intruded_jerk_std':[]}

        from tqdm import tqdm
        for trial in tqdm(range(k)):
            ob = self.env.reset(phase, max_epsilon=max_epsilon)

            done = False
            result = None
            states = []
            actions = []
            rewards = []
            epsilons = []
            robot_pos = []
            robot_vel = []
            human_vel = []
            perf_time = []
            dmins = []
            intruded_dmins = []
            times_intruded = 0
            absolute_minimum_distance = np.inf
            vel_intruded = []
            episode_collision_times = []
            episode_collision_pos = []
            episode_collision_blames = []

            while not done:
                perf_start = time.perf_counter()
                eps = self.env.get_epsilons(estimate_eps)
                action = self.robot.act(ob, eps)
                perf_end = time.perf_counter()
                perf_time.append(perf_end - perf_start)

                ob, reward, done, info = self.env.step(action)
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)
                epsilons.append(eps)

                robot_pos.append([self.robot.px, self.robot.py])
                robot_vel.append([self.robot.vx, self.robot.vy])
                for human in self.env.humans:
                    human_vel.append([human.vx, human.vy])

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist_danger.append(info.min_dist)

                if isinstance(info, HumanCollision):
                    h_pos = info.human.get_position()
                    h_vel = info.human.get_velocity()
                    r_pos = self.robot.get_position()
                    r_vel = self.robot.get_velocity()

                    episode_collision_times.append(self.env.global_time)
                    episode_collision_pos.append([round(coord,4) for coord in info.coll_pos])
                    episode_collision_blames.append(collision_blame(h_pos, h_vel, r_pos, r_vel))

                dmins.append(self.env.dmin)
                absolute_minimum_distance = min(absolute_minimum_distance, self.env.dmin)

                # record intrusions into personal space
                disc_dist = discomfort_dist(len(self.env.humans))
                if self.env.dmin < disc_dist:
                    times_intruded += 1
                    intruded_dmins.append(self.env.dmin)
                    vel_intruded.append([self.robot.vx, self.robot.vy])

            num_collisions = len(episode_collision_times)

            if num_collisions > 0:
                collision += 1
                collision_cases.append(trial)
                collision_times.append(episode_collision_times)
                result = 'HumanCollision'

            else:
                if isinstance(info, ReachGoal):
                    success += 1
                    success_times.append(self.env.global_time)
                    result = 'ReachGoal'
                elif isinstance(info, HumanCollision):
                    collision += 1
                    collision_cases.append(trial)
                    collision_times.append(self.env.global_time)
                    result = 'HumanCollision'
                elif isinstance(info, Timeout):
                    timeout += 1
                    timeout_cases.append(trial)
                    timeout_times.append(self.env.time_limit)
                    result = 'Timeout'
                else:
                    raise ValueError('Invalid end signal from environment')

            timeStep = self.env.time_step

            # Robot navigation metrics
            curves = []
            for i in range(len(robot_pos)-4):
                p0 = robot_pos[i]
                p1 = robot_pos[i+2]
                p2 = robot_pos[i+4]
                curves.append(np.abs(curvature(p0, p1, p2)))

            robot_vel = np.array(robot_vel)
            robotDF = pd.DataFrame(robot_vel, columns=['vx', 'vy'])
            robotDF[['ax', 'ay']] = robotDF[['vx', 'vy']].diff(4)#/timeStep
            robotDF[['jx', 'jy']] = robotDF[['ax', 'ay']].diff(4)#/timeStep
            robotDF['vel'] = np.sqrt(np.square(robotDF[['vx', 'vy']]).sum(axis=1))
            robotDF['acc'] = np.sqrt(np.square(robotDF[['ax', 'ay']]).sum(axis=1))
            robotDF['jer'] = np.sqrt(np.square(robotDF[['jx', 'jy']]).sum(axis=1))
            avg_robot_vel.append(np.mean(robotDF['vel']))
            avg_robot_acc.append(np.mean(robotDF['acc']))
            avg_robot_jer.append(np.mean(robotDF['jer']))

            # Human navigation metrics
            if human_vel:
                human_vel = np.array(human_vel)
                humanDF = pd.DataFrame(human_vel, columns=['vx', 'vy'])
                humanDF[['ax', 'ay']] = humanDF[['vx', 'vy']].diff(4)#/timeStep
                humanDF[['jx', 'jy']] = humanDF[['ax', 'ay']].diff(4)#/timeStep
                humanDF['vel'] = np.sqrt(np.square(humanDF[['vx', 'vy']]).sum(axis=1))
                humanDF['acc'] = np.sqrt(np.square(humanDF[['ax', 'ay']]).sum(axis=1))
                humanDF['jer'] = np.sqrt(np.square(humanDF[['jx', 'jy']]).sum(axis=1))
                avg_human_vel.append(np.mean(humanDF['vel']))
                avg_human_acc.append(np.mean(humanDF['acc']))
                avg_human_jer.append(np.mean(humanDF['jer']))

            if len(vel_intruded) > 0:
                vel_intruded = np.array(vel_intruded)
                intrudedDF = pd.DataFrame(vel_intruded, columns=['vx', 'vy'])
                intrudedDF[['ax', 'ay']] = intrudedDF[['vx', 'vy']].diff(4)#/timeStep
                intrudedDF[['jx', 'jy']] = intrudedDF[['ax', 'ay']].diff(4)#/timeStep
                intrudedDF['vel'] = np.sqrt(np.square(intrudedDF[['vx', 'vy']]).sum(axis=1))
                intrudedDF['acc'] = np.sqrt(np.square(intrudedDF[['ax', 'ay']]).sum(axis=1))
                intrudedDF['jer'] = np.sqrt(np.square(intrudedDF[['jx', 'jy']]).sum(axis=1))
                intrude_list = []
                intrude_list.append(np.mean(intrudedDF['vel']))  # intruded_vel_mean
                intrude_list.append(np.std(intrudedDF['vel']))   # intruded_vel_std
                intrude_list.append(np.mean(intrudedDF['acc']))  # intruded_acc_mean
                intrude_list.append(np.std(intrudedDF['acc']))   # intruded_acc_std
                intrude_list.append(np.mean(intrudedDF['jer']))  # intruded_jerk_mean
                intrude_list.append(np.std(intrudedDF['jer']))   # intruded_jerk_std
            else:
                intrude_list = [np.nan for _ in range(6)]

            # Record some metrics into the stats file
            stats['test_case'].append(trial)
            stats['scenario'].append(self.env.scenario)
            stats['perpetual'].append(self.env.perpetual)
            stats['num_humans'].append(len(self.env.humans))
            stats['open_space'].append(self.env.open_space)
            stats['navigation_time'].append(self.env.global_time)
            stats['result'].append(result)
            stats['num_collisions'].append(num_collisions)
            stats['collision_times'].append(episode_collision_times)
            stats['collision_positions'].append(episode_collision_pos)
            stats['collision_blames'].append(episode_collision_blames)
            stats['inference_time_mean'].append(np.mean(perf_time))
            stats['inference_time_std'].append(np.std(perf_time))
            stats['inference_time_min'].append(np.min(perf_time))
            stats['inference_time_max'].append(np.max(perf_time))
            stats['min_dist_mean'].append(np.mean(dmins))
            stats['min_dist_std'].append(np.std(dmins))
            stats['min_dist_min'].append(absolute_minimum_distance)
            stats['curvature_mean'].append(np.mean(curves))
            stats['curvature_std'].append(np.std(curves))
            stats['vel_mean'].append(np.mean(robotDF['vel']))
            stats['vel_std'].append(np.std(robotDF['vel']))
            stats['acc_mean'].append(np.mean(robotDF['acc']))
            stats['acc_std'].append(np.std(robotDF['acc']))
            stats['jerk_mean'].append(np.mean(robotDF['jer']))
            stats['jerk_std'].append(np.std(robotDF['jer']))
            stats['time_intruded'].append(timeStep*times_intruded)
            stats['intruded_min_dist_mean'].append(np.mean(intruded_dmins))
            stats['intruded_min_dist_std'].append(np.std(intruded_dmins))
            stats['intruded_vel_mean'].append(intrude_list[0])
            stats['intruded_vel_std'].append(intrude_list[1])
            stats['intruded_acc_mean'].append(intrude_list[2])
            stats['intruded_acc_std'].append(intrude_list[3])
            stats['intruded_jerk_mean'].append(intrude_list[4])
            stats['intruded_jerk_std'].append(intrude_list[5])

            if self.env.dmin != float('inf'):
                min_dist.append(self.env.dmin)

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, HumanCollision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, epsilons, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))

        stats_df = pd.DataFrame(stats)
        if self.stats_file is not None:
            stats_df.to_csv(self.stats_file, index=False)

        success_rate = success / k
        collision_rate = collision / k
        timeout_rate = timeout / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {} Success rate: {:.2f}, Collision rate: {:.2f}, Timeout rate: {:.2f}, Nav time: {:.2f}'.
                        format(phase.upper(), extra_info, success_rate, collision_rate, timeout_rate, avg_nav_time))

        if phase in ['val', 'test']:
            if min_dist:
                logging.info('Average minimum distance between robot and pedestrian: %.2f', average(min_dist))
            else:
                logging.info('Average minimum distance between robot and pedestrian: %.2f', -1)

            if min_dist_danger:
                logging.info('Average minimum distance when robot is too close: %.2f', average(min_dist_danger))
            else:
                logging.info('Average minimum distance when robot is too close: %.2f', -1)

            avg_avg_robot_vel = sum(avg_robot_vel) / len(avg_robot_vel)
            avg_avg_robot_acc = sum(avg_robot_acc) / len(avg_robot_acc)
            avg_avg_robot_jer = sum(avg_robot_jer) / len(avg_robot_jer)

            logging.info('Avg robot speed: {:.2f}, Avg robot accel: {:.2f}, Avg robot jerk: {:.2f}'.
                         format(avg_avg_robot_vel, avg_avg_robot_acc, avg_avg_robot_jer))

            if avg_human_vel:
                avg_avg_human_vel = sum(avg_human_vel) / len(avg_human_vel)
                avg_avg_human_acc = sum(avg_human_acc) / len(avg_human_acc)
                avg_avg_human_jer = sum(avg_human_jer) / len(avg_human_jer)
            else:
                avg_avg_human_vel = -1
                avg_avg_human_acc = -1
                avg_avg_human_jer = -1

            logging.info('Avg pedestrian speed: {:.2f}, Avg pedestrian accel: {:.2f}, Avg pedestrian jerk: {:.2f}'.
                         format(avg_avg_human_vel, avg_avg_human_acc, avg_avg_human_jer))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
            # logging.info('Collision locations: ' + ' '.join([str([round(x[0], 2), round(x[1], 2)]) for x in collision_positions]))

    def update_memory(self, states, actions, rewards, epsilons, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]
            epsilon = epsilons[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state, epsilon)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            # transform state of different human_num into fixed-size tensor
            if self.max_agents is not None and self.max_agents > 0:
                if len(state.size()) == 1:
                    human_num = 1
                    feature_size = state.size()[0]
                else:
                    human_num, feature_size = state.size()
                if human_num != self.max_agents:
                    padding = torch.zeros((self.max_agents - human_num, feature_size))
                    state = torch.cat([state, padding])
            self.memory.push((state, value))
