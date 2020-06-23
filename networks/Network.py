from abc import ABC, abstractmethod
import torch.nn as nn


class Network(nn.Module, ABC):
    def __init__(self, input_shape, n_actions):
        """ Base class to be derived by concrete network implementations

        Parameters
        -----------
        input_shape: array-like or tuple
            size of the input to the network
        n_actions: int
            number of possible actions in environment
        """
        super(Network, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

    @abstractmethod
    def forward(self, x):
        pass
