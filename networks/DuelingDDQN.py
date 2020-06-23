import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.Network import Network


class DuelingDDQN(Network):

    def __init__(self, input_shape, n_actions):
        """ Dueling Double Deep Q-Network (based on paper by Wang et al., 2016)
        Parameters
        -----------
        input_shape: array-like or tuple
            size of the input to the network
        n_actions: int
            number of possible actions in environment

        Attributes
        -----------
        conv1-3: torch.nn.Conv2d
            convolutional layers of the network
        fc_v: torch.nn.Linear
            fully-connnected layer for state value
        fc_a: torch.nn.Linear
            fully-connected layer for advantages
        v_stream: torch.nn.Linear
            fully-connected output layer for calculating state values
        f_stream: torch.nn.Linear
            fully-connected output layer for calculating advantages
        """
        super(DuelingDDQN, self).__init__(input_shape, n_actions)
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)

        fc_input = self.conv2d_size_out()

        self.fc_v = nn.Linear(in_features=fc_input, out_features=512)
        self.fc_a = nn.Linear(in_features=fc_input, out_features=512)
        self.v_stream = nn.Linear(in_features=512, out_features=1)
        self.a_stream = nn.Linear(in_features=512, out_features=n_actions)

    def forward(self, x):
        """ Calculates the forward pass in the network
        Parameters
        -----------
        x: nd-array
            mini-batch of state representations

        Returns
        -----------
        state_value: torch.tensor
            array of values for each state
        advantages: torch.tensor
            advantage values for every action for each state
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        state_value = self.v_stream(F.relu(self.fc_v(x)))
        advantages = self.a_stream(F.relu(self.fc_a(x)))
        return state_value, advantages

    def get_fc_input(self, input_shape):
        """ Calculates the input dims for fully-connected layer based on conv layers

        Parameters
        -----------
        input_shape: array-like or tuple
            size of the input to the network (before convolutions)

        Returns
        -----------
        fc_input: int
            size of data after conv layers
        """
        fc_input = self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape))))
        fc_input = int(torch.prod(fc_input))
        return fc_input
