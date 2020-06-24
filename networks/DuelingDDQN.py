import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from networks.Network import Network


class DuelingDDQN(Network):

    def __init__(self, input_shape, n_actions, device):
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
        fc_val: torch.nn.Linear
            fully-connnected layer for state value
        fc_adv: torch.nn.Linear
            fully-connected layer for advantages
        val_out: torch.nn.Linear
            fully-connected output layer for calculating state values
        adv_out: torch.nn.Linear
            fully-connected output layer for calculating advantages
        """
        super(DuelingDDQN, self).__init__(input_shape, n_actions)
        self.conv1 = nn.Conv2d(in_channels=self.input_shape[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)

        fc_input = self.get_fc_input()

        # self.fc_val = nn.Linear(in_features=fc_input, out_features=512)
        # self.fc_adv = nn.Linear(in_features=fc_input, out_features=512)
        self.fc1 = nn.Linear(fc_input, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.val_out = nn.Linear(in_features=512, out_features=1)
        self.adv_out = nn.Linear(in_features=512, out_features=n_actions)

        self.to(device)

    def forward(self, x):
        """ Calculates the forward pass in the network
        Parameters
        -----------
        x: nd-array
            mini-batch of state representations

        Returns
        -----------
        state_value: torch.tensor
            values for each state
        advantages: torch.tensor
            advantage values for every state,action pair
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        flat1 = F.relu(self.fc1(x))
        flat2 = F.relu(self.fc2(flat1))
        state_values = self.val_out(flat2)
        advantages = self.adv_out(flat2)

        # state_values = self.val_out(F.relu(self.fc_val(x)))
        # advantages = self.adv_out(F.relu(self.fc_adv(x)))
        # calc mean of every row
        mean_advantages = torch.mean(advantages, dim=1)
        # add second dimension
        mean_advantages = mean_advantages.unsqueeze(1)
        # equation (9) in paper
        ret = state_values + (advantages - mean_advantages)
        #print(ret)
        return ret


    def get_fc_input(self):
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
        fc_input = self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape))))
        fc_input = int(np.prod(fc_input.size()))
        return fc_input


if __name__ == '__main__':
    net = DuelingDDQN([1, 84, 84], 2)

    x = torch.rand([3, 1, 84, 84])
    net(x)


    print('succcces')
