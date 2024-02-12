import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('--plot_sr', default=False, action='store_true')
    parser.add_argument('--plot_cr', default=False, action='store_true')
    parser.add_argument('--plot_tr', default=False, action='store_true')
    parser.add_argument('--window_size', type=int, default=200)
    args = parser.parse_args()

    ax1_legends = ['Training', 'Validation']

    # Read output log file
    with open(args.model_dir + '/output.log', 'r') as file:
        log = file.read()

    val_pattern = r"VAL   in episode (?P<episode>\d+)  Success rate: (?P<sr>[0-1].\d+), " \
                    r"Collision rate: (?P<cr>[0-1].\d+), Timeout rate: (?P<tr>[0-1].\d+), Nav time: (?P<time>\d+.\d+)"
    val_episode = []
    val_sr, val_cr, val_tr = [], [], []
    for r in re.findall(val_pattern, log):
        val_episode.append(int(r[0]))
        val_sr.append(float(r[1]))
        val_cr.append(float(r[2]))
        val_tr.append(float(r[3]))

    train_pattern = r"TRAIN in episode (?P<episode>\d+)  Success rate: (?P<sr>[0-1].\d+), "\
                        r"Collision rate: (?P<cr>[0-1].\d+), Timeout rate: (?P<tr>[0-1].\d+), Nav time: (?P<time>\d+.\d+)"
    train_episode = []
    train_sr, train_cr, train_tr = [], [], []
    for r in re.findall(train_pattern, log):
        train_episode.append(int(r[0]))
        train_sr.append(float(r[1]))
        train_cr.append(float(r[2]))
        train_tr.append(float(r[3]))

    # Smooth training plots
    train_sr_smooth = running_mean(train_sr, args.window_size)
    train_cr_smooth = running_mean(train_cr, args.window_size)
    train_tr_smooth = running_mean(train_tr, args.window_size)

    # Plot success rate
    if args.plot_sr:
        _, ax1 = plt.subplots()
        ax1.plot(range(len(train_sr_smooth)), train_sr_smooth)
        ax1.plot(val_episode, val_sr)

        ax1.legend(ax1_legends)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rates Across Training Episodes')
        plt.savefig(args.model_dir + '/success.png')

    # Plot collision rate
    if args.plot_cr:
        _, ax1 = plt.subplots()
        ax1.plot(range(len(train_cr_smooth)), train_cr_smooth)
        ax1.plot(val_episode, val_cr)

        ax1.legend(ax1_legends)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Collision Rate')
        ax1.set_title('Collision Rates Across Training Episodes')
        plt.savefig(args.model_dir + '/collision.png')

    # Plot timeout rate
    if args.plot_tr:
        _, ax1 = plt.subplots()
        ax1.plot(range(len(train_tr_smooth)), train_tr_smooth)
        ax1.plot(val_episode, val_tr)

        ax1.legend(ax1_legends)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Timeout Rate')
        ax1.set_title('Timeout Rates Across Training Episodes')
        plt.savefig(args.model_dir + '/timeout.png')


if __name__ == '__main__':
    main()
