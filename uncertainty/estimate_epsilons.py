import os
import torch
import pickle
import numpy as np
import configparser

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

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

def get_Xy(positions, dt, time_size):
    # Obtain speed and velocity from pedestrian
    pos = np.array(positions).T
    vel = np.diff(pos) / dt
    acc = np.diff(vel) / dt
    speed = np.linalg.norm(vel, axis=0)[1:]
    accel = np.linalg.norm(acc, axis=0)

    # Compute the number of features in each sample
    sample_size = 2 * time_size + 1

    # Determine how many samples we can use
    num_samples = len(speed) - time_size + 1

    # Create an empty array to fill with samples
    X = np.zeros((num_samples, sample_size,))
    y = np.zeros((num_samples,))

    # Form each sample
    for a_sample in range(num_samples):
        # The features alternate speed and acceleration values
        for a_time in range(time_size):
            X[a_sample, a_time*2] = speed[a_sample + a_time]
            X[a_sample, a_time*2 + 1] = accel[a_sample + a_time]
    return X, y

def get_pred(X, y, model_dir):
    # Obtain model properties
    model_file = "{}/model.pth".format(model_dir)
    train_file = "{}/train.config".format(model_dir)
    train_config = configparser.RawConfigParser()
    train_config.read(train_file)
    hidden_dims = [int(x) for x in train_config.get('network', 'hidden_dims').split(', ')]
    if hidden_dims == [0]: hidden_dims = []
    nonlinearity = train_config.get('network', 'nonlinearity')

    # Normalize the input data
    scale_file = "{}/scaler.pkl".format(model_dir)
    with open(scale_file, 'rb') as scale_boi:
        scaler = pickle.load(scale_boi)
    X = scaler.transform(X)

    # Create a data loader
    test_data = Data(X, y)

    # Load in neural network model
    input_dim = X.shape[1]
    model = Net(input_dim, hidden_dims, nonlinearity)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # Make our predictions
    pred = model(test_data.X).detach().numpy()
    return np.clip(np.array(pred), 0, 1)

# Estimate uncertainty values from human position data
def estimate_epsilons(all_positions, dt):
    eps_pred = []
    for i, positions in enumerate(all_positions):
        time_size = np.minimum(20, len(positions) - 2)
        if time_size < 1:
            eps_pred.append(0.5)
        else:
            X, y = get_Xy(positions, dt, time_size)
            model_dir = '../uncertainty/models/uncertain_{}'.format(time_size)
            pred = np.mean(get_pred(X, y, model_dir))
            eps_pred.append(pred)
    return np.array(eps_pred)
