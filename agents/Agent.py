from abc import ABC, abstractmethod
import torch
import numpy as np


class Agent(ABC):
    def __init__(self, input_shape, n_actions, gamma=0.99, batch_size=128,
                 min_eps=0.01, max_eps=1, cutoff=1e6, device='GPU'):
        """ Base class to be derived by concrete agent implementations

        Parameters
        -----------
        input_shape: array-like or tuple
            size of the input to the network
        n_actions: int
            number of possible actions in environment
        gamma: float
            discount factor for future rewards
        batch_size: int
            number of observation for one forward and backward pass
        min_eps: float
            minimum value of epsilon (exploration ratio)
        max_eps: float
            maximum value of epsilon (exploration ratio)
        cutoff: float
            after how many steps to get to min_eps
        device: string
            device to run agent on (CPU or GPU)

        Attributes
        -----------
        step_counter: int
            how many steps in training so far
        """
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.possible_actions = np.arange(n_actions)
        self.gamma = gamma
        self.step_counter = 0
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.eps = self.max_eps
        self.cutoff = cutoff
        self.batch_size = batch_size

        if device.upper() == 'GPU':
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                print("Warning: GPU is not available, and CPU will be used instead!")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

    @abstractmethod
    def choose_action(self, state):
        pass

    def update_epsilon(self):
        """ Linearly decrease epsilon from max to min values
            so that it stays at min after cutoff
        """
        if self.eps <= self.min_eps:
            return

        self.eps = (self.min_eps - self.max_eps) * (self.step_counter / self.cutoff) + self.max_eps
