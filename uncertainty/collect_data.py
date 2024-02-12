import os
import gym
import time
import torch
import shutil
import logging
import argparse
import configparser
import numpy as np
import pandas as pd

from control.policy.policy_factory import policy_factory
from simulation.envs.utils.robot import Robot


def run_k_episodes(env, robot, k, offset, param):
    # Instantiate dicts to store our simulation data in.
    # 1) A dict to store all the positions of all the actors at every time step for each trial.
    positions_table = {"trial":[],
                       "time":[],
                       "robot":[],
                       "humans":[]}
    # 2) A dict to store the policy settings for each actor in each trial.
    agents_table = {"trial":[],
                    "actor":[],
                    "radius":[],
                    "v_pref":[],
                    "epsilon":[],
                    "start":[],
                    "goal":[],
                    "discomfort_dist":[],
                    "policy":[],
                    "policy_params":[]}
    # 3) A dict to store the environment configuration for each trial.
    scenarios_table = {"trial":[],
                       "scenario":[],
                       "perpetual":[],
                       "num_humans":[],
                       "room_size":[]}

    for i in range(k):
        done = False
        np.random.seed(i)
        ob = env.reset('test')
        if param is not None:
            if param == 'epsilon':
                for human in env.humans:
                    human.set_epsilon(np.random.uniform(0,1))
            elif param == 'speed':
                for human in env.humans:
                    human.set_policy_speed(np.random.uniform(0,2))
            elif param == 'horizon':
                for human in env.humans:
                    human.set_policy_horizon(np.random.uniform(0,3))
            elif param == 'safety':
                for human in env.humans:
                    human.set_policy_safety(np.random.uniform(0,1))
            elif param == 'neighbors':
                for human in env.humans:
                    human.set_policy_neighbors(round(np.random.uniform(0,20)))
            else:
                raise ValueError('Unknown human parameter to vary')

        # Store environment configuration for this trial.
        scenarios_table["trial"].append(i + offset)
        scenarios_table["scenario"].append(env.scenario)
        scenarios_table["perpetual"].append(env.perpetual)
        scenarios_table["num_humans"].append(len(env.humans))
        scenarios_table["room_size"].append(env.room_dims)

        # Store robot policy settings for this trial.
        agents_table["trial"].append(i + offset)
        agents_table["actor"].append("robot")
        agents_table["radius"].append(env.robot.radius)
        agents_table["v_pref"].append(env.robot.v_pref)
        agents_table["epsilon"].append(0)
        agents_table["start"].append(env.robot.get_position())
        agents_table["goal"].append(env.robot.get_goal_position())
        agents_table["discomfort_dist"].append(env.discomfort_dist)
        agents_table["policy"].append(env.robot.policy.name)
        agents_table["policy_params"].append(env.robot.policy.get_params())

        # Store human policy settings for this trial.
        agents_table["trial"].append(i + offset)
        agents_table["actor"].append("humans")
        temp_dict = {"radius":[],
                     "v_pref":[],
                     "epsilon":[],
                     "start":[],
                     "goal":[],
                     "discomfort_dist":[],
                     "policy":[],
                     "policy_params":[]}
        for aHuman in env.humans:
            temp_dict["radius"].append(aHuman.radius)
            temp_dict["v_pref"].append(aHuman.v_pref)
            temp_dict["epsilon"].append(aHuman.epsilon)
            temp_dict["start"].append(aHuman.get_position())
            temp_dict["goal"].append(aHuman.get_goal_position())
            temp_dict["discomfort_dist"].append(env.discomfort_dist) # !! This might change to be agent-specific
            temp_dict["policy"].append(aHuman.policy.name)
            temp_dict["policy_params"].append(aHuman.policy.get_params())
        for aKey in temp_dict:
            agents_table[aKey].append(temp_dict[aKey])

        # Store first time step of positions for our agents.
        positions_table["trial"].append(i + offset)
        positions_table["time"].append(env.global_time)
        positions_table["robot"].append(robot.get_position())
        temp_list = []
        for aHuman in env.humans:
            temp_list.append(aHuman.get_position())
        positions_table["humans"].append(temp_list)

        while not done:
            action = robot.act(ob)
            ob, reward, done, info = env.step(action)

            # Record the positions of our agents.
            positions_table["trial"].append(i + offset)
            positions_table["time"].append(env.global_time)
            positions_table["robot"].append(robot.get_position())
            temp_list = []
            for aHuman in env.humans:
                temp_list.append(aHuman.get_position())
            positions_table["humans"].append(temp_list)

    # Convert our dictionaries to pandas dataframes
    positions_df = pd.DataFrame(positions_table)
    agents_df = pd.DataFrame(agents_table)
    scenarios_df = pd.DataFrame(scenarios_table)
    return positions_df, agents_df, scenarios_df


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='data/')
    parser.add_argument('--env_config', type=str, default='../control/configs/env.config')
    parser.add_argument('--policy_config', type=str, default='../control/configs/policy.config')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--vary_param', type=str, default=None)
    args = parser.parse_args()

    # Read model weights
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

    # Configure paths
    make_new_dir = True
    if os.path.exists(args.output_dir):
        key = input('Output directory already exists! Overwrite the folder? (y/n)')
        if key == 'y':
            shutil.rmtree(args.output_dir)
        else:
            make_new_dir = False
            args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
    if make_new_dir:
        os.makedirs(args.output_dir)

    log_file = os.path.join(args.output_dir, 'output.log')
    agents_file = os.path.join(args.output_dir, 'agents.csv')
    scenarios_file = os.path.join(args.output_dir, 'scenarios.csv')
    positions_file = os.path.join(args.output_dir, 'positions.csv')

    # Configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S", filename=log_file, filemode='a')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Using device: %s', device)

    # Configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights))

    # Configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    robot.policy.set_phase('test')
    env.set_robot(robot)

    # Initialize policy
    policy.set_phase('test')
    policy.set_device(device)
    policy.set_env(env)
    robot.print_info()

    # Run n trials in each scenario
    trials, offset = args.num_episodes, 0
    scenarios  = ['circle', 'perpedendicular', 'opposite', 'same', 'random', 'random']
    perpetuals = [False, False, False, False, False, True]
    positions_frames, agents_frames, scenarios_frames = [], [], []
    for scenario, perpetual in zip(scenarios, perpetuals):
        print("Running {} trials of {} scenario with perpetual = {}".format(trials, scenario, perpetual))
        env.scenario, env.perpetual = scenario, perpetual
        positions_frame, agents_frame, scenarios_frame = run_k_episodes(env, robot, trials, offset, args.vary_param)
        positions_frames += [positions_frame]
        agents_frames += [agents_frame]
        scenarios_frames += [scenarios_frame]
        offset += trials

    # Save pedestrian data to csv
    positions_df = pd.concat(positions_frames)
    agents_df = pd.concat(agents_frames)
    scenarios_df = pd.concat(scenarios_frames)
    positions_df.to_csv(positions_file, index=False)
    agents_df.to_csv(agents_file, index=False)
    scenarios_df.to_csv(scenarios_file, index=False)


if __name__ == '__main__':
    main()
