import os
import gym
import torch
import logging
import argparse
import configparser
import numpy as np
import numpy.linalg as la

from control.utils.explorer import Explorer, collision_blame
from control.policy.policy_factory import policy_factory
from simulation.envs.utils.robot import Robot
from simulation.envs.policy.orca import ORCA
from simulation.envs.utils.info import *


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--max_epsilon', type=float, default=0.0)
    parser.add_argument('--num_episodes', type=float, default=0.0)
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--stats_file', type=str, default='stats.csv')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--estimate_eps', default=False, action='store_true')
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    args = parser.parse_args()

    # configure logging and device
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
    #                     datefmt="%Y-%m-%d %H:%M:%S", filename='thicc_boi.log', filemode='a')
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    # logging.info('Using device: %s', device)

    # read config files
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

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    # initialize policy
    policy.set_phase(args.phase)
    policy.set_device(device)
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            robot.policy.safety_space = 0
        # logging.info('ORCA agent buffer: %f', robot.policy.safety_space)
    policy.set_env(env)
    robot.print_info()

    # visualize evaluation
    if args.visualize:
        ob = env.reset(phase=args.phase, test_case=args.test_case, max_epsilon=args.max_epsilon)
        done = False
        last_pos = np.array(robot.get_position())
        while not done:
            eps = env.get_epsilons(args.estimate_eps)
            action = robot.act(ob, eps)
            ob, _, done, info = env.step(action)
            current_pos = np.array(robot.get_position())
            # logging.debug('Speed: %.2f', la.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos

        print('Testing: {} scenario with {} pedestrians'.format(env.scenario, len(env.humans)))
        print('Result:  {}'.format(info))

        if args.traj:
            env.render('traj', args.video_file)
        else:
            env.render('video', args.video_file)

        # logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        # if robot.visible and info == 'reach goal':
        #     human_times = env.get_human_times()
        #     logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))

    # run evaluation without visualization
    else:
        explorer = Explorer(env, robot, device, gamma=0.9, stats_file=args.stats_file)
        k = int(args.num_episodes if args.num_episodes > 0 else env.case_size[args.phase])
        explorer.run_k_episodes(k, args.phase, print_failure=True, max_epsilon=args.max_epsilon, estimate_eps=args.estimate_eps)


if __name__ == '__main__':
    main()
