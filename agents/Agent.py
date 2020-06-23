from abc import ABC, abstractmethod
import torch
import numpy as np


class Agent(ABC):
    def __init__(self, input_shape, n_actions, optimizer, gamma=0.99, C=4,
                 min_eps=0.01, max_eps=1, device='GPU', checkpoint_path='scores'):
        """ Base class to be derived by concrete agent implementations

        Parameters
        -----------
        input_shape: array-like or tuple
            size of the input to the network
        n_actions: int
            number of possible actions in environment
        optimizer: torch.optimizer
            optimizer for loss and weight updates
        gamma: float
            discount factor for future rewards
        C : int
            how many steps to take before replacing target network
        min_eps: float
            minimum value of epsilon (exploratio)
        max_eps: float
            maximum value of epsilon
        device: string
            device to run training on (CPU or GPU)
        checkpoint_path: string
            where to save the model

        Attributes
        -----------
        C_counter: int
            how many steps since last target network swap
        step_counter: int
            how many steps in training so far
        """
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.possible_actions = np.arange(n_actions)
        self.optimizer = optimizer
        self.gamma = gamma
        self.C = C
        self.C_counter = 0
        self.step_counter = 0
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.checkpoint_path = checkpoint_path

        if device.upper() == 'GPU' and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif device.upper() == 'GPU':
            print("Warning: GPU is not available, and CPU will be used instead!")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def calculate_loss(self):
        pass

    @abstractmethod
    def back_propagate(self):
        pass

    def update_epsilon(self):
        pass
