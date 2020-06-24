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

        conv_out = self.get_conv_out()

        self.fc_val = nn.Linear(in_features=conv_out, out_features=512)
        self.fc_adv = nn.Linear(in_features=conv_out, out_features=512)
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

        # calculate outputs of each stream
        state_values = self.val_out(F.relu(self.fc_val(x)))
        advantages = self.adv_out(F.relu(self.fc_adv(x)))

        # calculate advantage mean of every row
        mean_advantages = torch.mean(advantages, dim=1).unsqueeze(1)

        # equation (9) in paper describing network output
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s))
        ret = state_values + (advantages - mean_advantages)
        return ret


    def get_conv_out(self):
        """ Calculates the output dims of convolutional layers
        Returns
        -----------
        fc_input: int
            size of data after conv layers
        """
        conv_out = self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape))))
        conv_out = int(np.prod(conv_out.size()))
        return conv_out


if __name__ == '__main__':
    net = DuelingDDQN([1, 84, 84], 2)

    x = torch.rand([3, 1, 84, 84])
    net(x)


    print('succcces')
