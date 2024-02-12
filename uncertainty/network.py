import torch
from torch import nn


# Multilayer perceptron (MLP)
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dims, nonlinearity):
        super().__init__()
        layers = []
        mlp_dims = [input_dim] + hidden_dims + [1]
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                if nonlinearity == 'relu':
                    layers.append(nn.ReLU())
                elif nonlinearity == 'leaky':
                    layers.append(nn.LeakyReLU())
                elif nonlinearity == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif nonlinearity == 'tanh':
                    layers.append(nn.Tanh())
                elif nonlinearity == 'gelu':
                    layers.append(nn.GELU())
                else:
                    print('Warning: unknown activation function')
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        z = self.net(x)
        return z.flatten()
