import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DuelingDQN(nn.Module):

    def __init__(self, input_shape, n_actions):
        """
        Parameters
        -----------
        input_shape: array-like
            size of the input to the network
        lr: float
            learning rate
        n_actions:
            number of possible actions in environment

        """
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)

        fc_input = self.conv2d_size_out()

        self.fc = nn.Linear(in_features=fc_input, out_features=512)
        self.v_stream = nn.Linear(in_features=512, out_features=1)
        self.a_stream = nn.Linear(in_features=512, out_features=n_actions)

    def forward(self, x):
        """ Calculates the forward pass in the network
        Parameters
        -----------
        x: nd-array
            representation of one state in the game

        Returns
        -----------
        state_value: int
            value of the state
        advantages: array-like
            advantage value for every action
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        state_value = self.v_stream(x)
        advantages = self.a_stream(x)
        return state_value, advantages

    def get_fc_input(self, input_shape):
        """ Calculates the input dims for fully-connected layer based on conv layers

        Parameters
        -----------
        input_shape: array-like
            size of the input to the network (before convolutions)

        Returns
        -----------
        fc_input: int
            size of data after conv layers
        """
        fc_input = self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape))))
        fc_input = int(torch.prod(fc_input))
        return fc_input
