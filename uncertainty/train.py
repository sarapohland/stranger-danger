import os
import ast
import sys
import torch
import shutil
import pickle
import logging
import argparse
import numpy as np
import configparser
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

from uncertainty.network import Net


class Data(Dataset):
  def __init__(self, X_train, y_train):
    self.X = torch.from_numpy(X_train.astype(np.float32))
    self.y = torch.from_numpy(y_train.astype(np.float32))
    self.len = self.X.shape[0]

  def __getitem__(self, index):
    return self.X[index], self.y[index]

  def __len__(self):
    return self.len

# Convert strings in a DataFrame into lists, tuples, and dicts as necessary
def convert_columns(dfBoi):
    brackets = set(['[', '{', '('])
    for aKey in dfBoi:
        dfBoi[aKey] = dfBoi[aKey].apply(lambda x: ast.literal_eval(str(x)) if str(x)[0] in brackets else x)

def extract_data(data_dir, time_size, step_size, labels, num_neighbors, density_radii):
    # Load in our collected and preprocessed data, and convert lists accordingly
    scenarios_df = pd.read_csv("{}/scenarios.csv".format(data_dir))
    agents_df = pd.read_csv("{}/agents.csv".format(data_dir))
    df_list = [scenarios_df, agents_df]
    for aDF in tqdm(df_list):
        convert_columns(aDF)

    # Instantiate our resulting matrices X and y
    X = []
    y = []

    # Calculate the length of each sample
    block_size = time_size*len(labels)
    sample_size = block_size + num_neighbors + len(density_radii)

    # Iterate through all of our trials.
    trial_list = scenarios_df['trial'].unique()
    for a_trial in tqdm(trial_list):
        # Grab the epsilon values for all the humans in this trial
        human_epsilons = agents_df['epsilon'].loc[(agents_df['trial'] == a_trial) & (agents_df['actor'] == 'humans')].iloc[0]

        # Iterate through each human in this trial
        for a_human in range(scenarios_df.loc[scenarios_df['trial'] == a_trial]['num_humans'].iloc[0]):
            # Load in the human data and convert lists accordingly
            human_df = pd.read_csv("{}/preprocessed_data/trial_{}/{}.csv".format(data_dir, a_trial, a_human))
            convert_columns(human_df)

            # Calculate the number of samples we can extract from this human
            num_rows = len(human_df) - 2
            num_samples = 0 if num_rows < time_size else (num_rows - time_size)//step_size + 1

            # Generate each sample
            for sample_num in range(num_samples):
                # Instantiate a numpy array to hold this sample
                a_sample = np.zeros(sample_size)

                # Populate our array values
                start_row = 2 + sample_num*step_size
                a_sample[:block_size] = human_df[labels].iloc[start_row:start_row + time_size].to_numpy().flatten()
                for a_neighbor in range(min(num_neighbors, len(human_df['neighbor_dists'][start_row]) - 1)):
                    a_sample[block_size + a_neighbor] = 1/human_df['neighbor_dists'][start_row][a_neighbor+1]
                for rad_num, a_radius in enumerate(density_radii):
                    a_sample[block_size + num_neighbors + rad_num] = sum([a_dist < a_radius for a_dist in human_df['neighbor_dists'][start_row]])

                X.append(a_sample)
                y.append(human_epsilons[a_human])

    # Convert X and y into numpy arrays before returning
    X = np.array(X)
    y = np.array(y)

    return X, y

def train(dataloader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0
    size = len(dataloader.dataset)

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = train_loss / len(dataloader)
    logging.info('Average training loss: %f', avg_loss)
    return avg_loss

def test(dataloader, model, criterion, device):
    model.eval()
    test_loss = 0
    size = len(dataloader.dataset)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item()

    avg_loss = test_loss / len(dataloader)
    logging.info('Average test loss: %f', avg_loss)
    return avg_loss

def plot_epsilons(dataloader, model, device, fig_file, args):
    model.eval()

    preds = []
    ys = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X).detach().cpu().numpy()
            preds.append(pred)
            ys.append(y.detach().cpu().numpy())
    pred = np.hstack(preds)
    y = np.hstack(ys)

    # Sort our results by target epsilon
    results = np.vstack([y, pred]).T
    results = results[results[:, 0].argsort()]

    # Generate some mean/std and quartiles for plotting
    targets = []
    means = []
    stds = []
    medians = []
    firstQs = []
    thirdQs = []

    cur_ind = 0
    while cur_ind < results.shape[0]:
        cur_epsilon = results[cur_ind, 0]
        mask = results[:,0] == cur_epsilon
        cur_preds = results[:,1][mask]

        targets.append(cur_epsilon)
        means.append(np.mean(cur_preds))
        stds.append(np.std(cur_preds))
        percentiles = np.percentile(cur_preds, [25, 50, 75])
        firstQs.append(percentiles[0])
        medians.append(percentiles[1])
        thirdQs.append(percentiles[2])

        cur_ind += len(cur_preds)

    # Smooth our data
    window = (len(targets)//20)*2 + 1 # Some reasonably odd number
    poly_order = 3

    targets = savgol_filter(np.array(targets), window, poly_order)
    means = savgol_filter(np.array(means), window, poly_order)
    stds = savgol_filter(np.array(stds), window, poly_order)
    medians = savgol_filter(np.array(medians), window, poly_order)
    firstQs = savgol_filter(np.array(firstQs), window, poly_order)
    thirdQs = savgol_filter(np.array(thirdQs), window, poly_order)

    # Plot and save our figure
    plt.figure(figsize=(8, 8))
    plt.title("Evaluation of {}\non test split of {}".format(args.output_dir, args.data_dir))
    plt.xlabel("target epsilon")
    plt.ylabel("predicted epsilon")
    plt.plot(targets, means, 'b-', label="mean +/- std")
    plt.fill_between(targets, means - stds, means + stds, color='b', alpha=0.2)
    plt.plot(targets, medians, "r-", label="median/quartiles")
    plt.fill_between(targets, firstQs, thirdQs, color='r', alpha=0.2)
    plt.plot([0,1], [0,1], color='k', label='y=x')
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.savefig(fig_file, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--output_dir', type=str, default='model/')
    parser.add_argument('--train_config', type=str, default='configs/train.config')
    args = parser.parse_args()

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
        shutil.copy(args.train_config, args.output_dir + '/train.config')

    log_file   = os.path.join(args.output_dir, 'output.log')
    scale_file = os.path.join(args.output_dir, 'scaler.pkl')
    model_file = os.path.join(args.output_dir, 'model.pth')
    loss_file  = os.path.join(args.output_dir, 'loss.png')
    fig_file   = os.path.join(args.output_dir, 'epsilons_test.png')

    # Configure logging and device
    file_handler = logging.FileHandler(log_file)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S", filename=log_file, filemode='a')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Using device: %s', device)

    # Read training parameters
    if args.train_config is None:
        parser.error('Train config file has to be specified.')
    train_config = configparser.RawConfigParser()
    train_config.read(args.train_config)
    time_size = train_config.getint('features', 'time_size')
    step_size = train_config.getint('features', 'step_size')
    labels = train_config.get('features', 'labels').split(', ')
    num_neighbors = train_config.getint('features', 'num_neighbors')
    density_radii = [float(x) for x in train_config.get('features', 'density_radii').split(', ')]
    hidden_dims = [int(x) for x in train_config.get('network', 'hidden_dims').split(', ')]
    if hidden_dims == [0]: hidden_dims = []
    nonlinearity = train_config.get('network', 'nonlinearity')
    optimizer = train_config.get('optimizer', 'optimizer')
    learning_rate = train_config.getfloat('optimizer', 'learning_rate')
    momentum = train_config.getfloat('optimizer', 'momentum')
    betas = [float(x) for x in train_config.get('optimizer', 'betas').split(', ')]
    epsilon = train_config.getfloat('optimizer', 'epsilon')
    weight_decay = train_config.getfloat('optimizer', 'weight_decay')
    batch_size = train_config.getint('training', 'batch_size')
    epochs = train_config.getint('training', 'epochs')

    # Extract pedestrian data for training
    X, y = extract_data(args.data_dir, time_size, step_size, labels, num_neighbors, density_radii) # TODO: MAY NEED TO UPDATE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize input data
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    with open(scale_file, "wb") as output_file:
        pickle.dump(scaler, output_file)

    # Create data loaders
    train_data = Data(X_train, y_train)
    test_data  = Data(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader  = DataLoader(test_data, batch_size=batch_size)

    # Define neural network model
    input_dim = X_train.shape[1]
    model = Net(input_dim, hidden_dims, nonlinearity).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                        momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                        betas=betas, eps=epsilon, weight_decay=weight_decay)
    else:
        raise ValueError('Optimizers: sgd, adam')

    # Run training process over several epochs
    train_loss = []
    test_loss  = []
    for t in range(epochs):
        logging.info('Epoch %d -------------------------------', t+1)
        train_loss += [train(train_loader, model, criterion, optimizer, device)]
        test_loss  += [test(test_loader, model, criterion, device)]

    # Save plot of loss over epochs
    plt.plot(train_loss, '-b', label='Training')
    plt.plot(test_loss, '-r', label='Evaluation')
    plt.legend(loc="upper right")
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss Across Batches')
    plt.title('Average Training and Evaluation Loss')
    plt.savefig(loss_file)

    # Save trained model
    torch.save(model.state_dict(), model_file)

    # Evaluate and plot epsilon values
    plot_epsilons(test_loader, model, device, fig_file, args)

if __name__ == '__main__':
    main()
