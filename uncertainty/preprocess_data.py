import os
import ast
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--data_dir', type=str, default='data/')
    args = parser.parse_args()

    # Load in our collected data
    data_dir = args.data_dir
    scenarios_df = pd.read_csv("{}/scenarios.csv".format(data_dir))
    agents_df = pd.read_csv("{}/agents.csv".format(data_dir))
    positions_df = pd.read_csv("{}/positions.csv".format(data_dir))
    df_list = [scenarios_df, agents_df, positions_df]

    # Convert strings into lists, tuples, and dicts as necessary
    def convert_columns(dfBoi):
        brackets = set(['[', '{', '('])
        for aKey in dfBoi:
            dfBoi[aKey] = dfBoi[aKey].apply(lambda x: ast.literal_eval(str(x)) if str(x)[0] in brackets else x)

    for aDF in tqdm(df_list):
        convert_columns(aDF)

    # Helper functions for curvature
    def get_curvature(p0, p1, p2):
        area = _get_area(p0, p1, p2)
        d0 = _get_dist(p0, p1)
        d1 = _get_dist(p1, p2)
        d2 = _get_dist(p2, p0)
        return 4*area/(d0*d1*d2)

    def _get_area(p0, p1, p2):
        return (p1[0] - p0[0])*(p2[1] - p0[1]) - (p1[1] - p0[1])*(p2[0] - p0[0])

    def _get_dist(p0, p1):
        return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

    # Define a folder to store our results in
    results_dir = "{}/preprocessed_data".format(data_dir)

    # Iterate through all the trials and generate data for each human
    trial_list = scenarios_df['trial'].unique()
    for a_trial in tqdm(trial_list):
        # Make a folder to store the results for each trial in
        save_dir = "{}/trial_{}".format(results_dir, a_trial)
        os.makedirs(save_dir, exist_ok=True)

        # Grab the timestamps for this trial
        times = positions_df.loc[positions_df['trial'] == a_trial]['time']
        dts = times.diff()

        # Grab human position values for this trial
        positions = positions_df.loc[positions_df['trial'] == a_trial]
        humans_xpos = []
        humans_ypos = []
        for aRow in range(len(positions)):
            human_pos = np.array(positions['humans'].iloc[aRow]).T
            humans_xpos.append(human_pos[0])
            humans_ypos.append(human_pos[1])
        humans_xpos = np.array(humans_xpos)
        humans_ypos = np.array(humans_ypos)

        # Grab robot position values for this trial
        robot_pos = np.array(tuple(positions['robot']))

        # Generate additional data for each human
        for a_human in range(scenarios_df.loc[scenarios_df['trial'] == a_trial]['num_humans'].iloc[0]):
            # If we've already calculated values for this human, we can skip it
            human_file = "{}/{}.csv".format(save_dir, a_human)
            if os.path.exists(human_file):
                continue

            # Times and positions
            human_df = pd.DataFrame({'time' : times,
                                     'dt' : dts,
                                     'xpos' : humans_xpos[:,a_human],
                                     'ypos' : humans_ypos[:,a_human]})

            # Velocity and acceleration (vector)
            human_df['xvel'] = human_df['xpos'].diff()/human_df['dt']
            human_df['xacc'] = human_df['xvel'].diff()/human_df['dt']
            human_df['yvel'] = human_df['ypos'].diff()/human_df['dt']
            human_df['yacc'] = human_df['yvel'].diff()/human_df['dt']

            # Distance from the robot
            human_df['dist'] = np.sqrt(np.square(human_df[['xpos', 'ypos']] - robot_pos).sum(axis=1))

            # Speed and acceleration (scalar)
            human_df['speed'] = np.sqrt(np.square(human_df[['xvel', 'yvel']]).sum(axis=1))
            human_df['accel'] = np.sqrt(np.square(human_df[['xacc', 'yacc']]).sum(axis=1))

            # Curvature
            curves = [0, 0]
            p1 = [human_df['xpos'].iloc[0], human_df['ypos'].iloc[0]]
            p0 = [human_df['xpos'].iloc[1], human_df['ypos'].iloc[1]]
            for curve_row in range(2, len(human_df)):
                p2 = p1
                p1 = p0
                p0 = [human_df['xpos'].iloc[curve_row], human_df['ypos'].iloc[curve_row]]
                curves.append(get_curvature(p0, p1, p2))
            human_df['curve'] = curves

            # Distances to other people
            neighbor_dists = []
            for curve_row in range(len(human_df)):
                p0 = np.array([[human_df['xpos'].iloc[curve_row], human_df['ypos'].iloc[curve_row]]])
                other_pos = np.array([humans_xpos[curve_row,:], humans_ypos[curve_row,:]]).T
                cur_dists = np.sort(np.sqrt(np.square(p0 - other_pos).sum(axis=1)))
                neighbor_dists.append(list(cur_dists))
            human_df['neighbor_dists'] = neighbor_dists

            # Angular acceleration
            ang_acc = [0]
            for ang_row in range(len(human_df)-1):
                dot_prod = human_df['xvel'].iloc[ang_row] * human_df['xvel'].iloc[ang_row+1] + human_df['yvel'].iloc[ang_row] * human_df['yvel'].iloc[ang_row+1]
                mag_prod = human_df['speed'].iloc[ang_row] * human_df['speed'].iloc[ang_row+1]
                ang_acc.append(np.arccos(dot_prod/mag_prod)/human_df['dt'].iloc[ang_row+1])
            human_df['ang_acc'] = ang_acc

            # Linear acceleration (i.e. difference in speed)
            human_df['lin_acc'] = human_df['speed'].diff()/human_df['dt']

            # Save our calculations to a file
            human_df.to_csv(human_file)

if __name__ == '__main__':
    main()
