import math
import argparse
import numpy as np
import pandas as pd

from ast import literal_eval
from tabulate import tabulate
from matplotlib import pyplot as plt


def get_result_string(a_series):
    a_result = "{:.2f}".format(np.mean(a_series))
    for a_quant in [0.9, 0.95]:
        # Quantile (i.e. Value at risk.)
        #a_result += "/{:.3f}".format(a_series.quantile(a_quant))

        # Expected value above the quantile (i.e. Conditional Value at Risk)
        a_result += "/{:.2f}".format(np.mean(a_series.loc[(a_series >= a_series.quantile(a_quant))]))
    return a_result

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--files', action='append', nargs='+')
    parser.add_argument('--names', action='append', nargs='+')
    args = parser.parse_args()

    metric_files = args.files[0]
    policy_names = args.names[0]

    # Ignore collisions within the first time step
    metrics_dfs = []
    for metric_file in metric_files:
        metric_df = pd.read_csv(metric_file)
        for index, row in metric_df.iterrows():
            collision_times = literal_eval(row['collision_times'])
            collision_positions = literal_eval(row['collision_positions'])
            collision_blames = literal_eval(row['rob_collision_blames'])
            metric_df.at[index, 'collision_times'] = []
            metric_df.at[index, 'collision_positions'] = []
            metric_df.at[index, 'rob_collision_blames'] = []
            for j, collision_time in enumerate(collision_times):
                if collision_time > 0.5:
                    metric_df.at[index, 'collision_times'].append(collision_time)
                    metric_df.at[index, 'collision_positions'].append(collision_positions[j])
                    metric_df.at[index, 'rob_collision_blames'].append(collision_blames[j])
            num_collisions = len(metric_df.at[index, 'collision_times'])
            metric_df.at[index, 'num_collisions'] = num_collisions
            if num_collisions == 0 and row['result'] == 'HumanCollision':
                metric_df.at[index, 'result'] = 'Timeout' if row['navigation_time'] > 29.0 else 'ReachGoal'
        metrics_dfs.append(metric_df)

    # Calculate path lengths
    for metrics_df in metrics_dfs:
        metrics_df['path_length'] = metrics_df['navigation_time'] * metrics_df['vel_mean']

    # Define nominal path length and navigation time
    V_PREF = 1 # m/s
    ROOM_HEIGHT = 15 # meters
    GOAL_RADIUS = 0.3 # meters
    ROBOT_RADIUS = 0.3 # meters
    OPT_PATH_LENGTH = 2 * (ROOM_HEIGHT/2 - 1) - GOAL_RADIUS - ROBOT_RADIUS
    OPT_NAV_TIME = OPT_PATH_LENGTH / V_PREF

    results = [[name] for name in policy_names]
    for policy_index, metrics_df in enumerate(metrics_dfs):
        # success/collision/timeout rates
        total_trials = len(metrics_df)
        results[policy_index].append(sum(metrics_df['result'] == 'ReachGoal') / total_trials * 100) # Success Rate
        results[policy_index].append(sum(metrics_df['result'] == 'Timeout') / total_trials * 100) # Timeout Rate
        results[policy_index].append(sum(metrics_df['result'] == 'HumanCollision') / total_trials * 100) # Collision Rate

        # Only consider successful runs so we don't skew our navigation times or
        # minimum distance values with collisions and timeouts
        smol_metrics = metrics_df.loc[(metrics_df['result'] == 'ReachGoal') | (metrics_df['result'] == 'HumanCollision')]

        # normalized navigation time
        normed_times = smol_metrics['navigation_time'] / OPT_NAV_TIME

        results[policy_index].append(get_result_string(normed_times))
        # results[policy_index].append(np.mean(normed_times))
        # results[policy_index].append("{:.4f} \u00B1 {:.4f}".format(np.mean(normed_times), np.std(normed_times)))

        # normalized path lengths
        normed_lengths = smol_metrics['path_length'] / OPT_PATH_LENGTH
        results[policy_index].append(get_result_string(normed_lengths))
        # results[policy_index].append(np.mean(normed_lengths))
        # results[policy_index].append("{:.4f} \u00B1 {:.4f}".format(np.mean(normed_lengths), np.std(normed_lengths)))

        # total number of collisions between the robot and a pedestrian across all trials
        sum_collisions = sum(metrics_df['num_collisions'])
        results[policy_index].append(sum_collisions)

        # personal space cost
        costs = metrics_df['avg_max_cost']*1000
        results[policy_index].append(get_result_string(costs))
        # results[policy_index].append(np.mean(costs))
        # results[policy_index].append("{:.4f} \u00B1 {:.4f}".format(np.mean(costs), np.std(costs)))

        # proportion of time spent in someone's personal space
        intruded_props = metrics_df['pers_time_intruded'] / metrics_df['navigation_time'] * 100
        results[policy_index].append(get_result_string(intruded_props))
        # results[policy_index].append(np.mean(intruded_props))
        # results[policy_index].append("{:.4f} \u00B1 {:.4f}".format(np.mean(intruded_props), np.std(intruded_props)))

        # proportion of time spent in someone's intimate space
        intruded_props = metrics_df['int_time_intruded'] / metrics_df['navigation_time'] * 100
        results[policy_index].append(get_result_string(intruded_props))
        # results[policy_index].append(np.mean(intruded_props))
        # results[policy_index].append("{:.4f} \u00B1 {:.4f}".format(np.mean(intruded_props), np.std(intruded_props)))

        # # "accountability" of the robot when a collision occurs
        # collision_blames = []
        # for row in metrics_df['rob_collision_blames']:
        #   if row:
        #     collision_blames += [val for val in row]
        # if collision_blames:
        #   avg_accountability = np.mean(collision_blames)
        #   std_accountability = np.std(collision_blames)
        # results[policy_index].append(avg_accountability)
        # # results[policy_index].append("{:.4f} \u00B1 {:.4f}".format(avg_accountability, std_accountability))

    # Print the table so it looks nice-ish
    headers = ['Navigation Policy', 'Success Rate', 'Timeout Rate', 'Collision Rate',
            'Relative Navigation Time', 'Relative Path Length', 'Number of Collisions',
            'Personal Space Cost', 'Personal Space Violation', 'Intimate Space Violation']

    # print(tabulate(results, headers=headers, tablefmt="github", numalign='center', floatfmt=".4f"))
    print(tabulate(results, headers=headers, tablefmt="latex", numalign='center', floatfmt=".3f"))


if __name__ == '__main__':
    main()
