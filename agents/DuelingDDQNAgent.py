from ReplayMemory import ReplayMemory
from agents.Agent import Agent
from networks.DuelingDDQN import DuelingDDQN


class DuelingDDQNAgent(Agent):
    """ Dueling Double Deep Q-Network Agent

    Attributes
    -----------
    memory: ReplayMemory
        memory of the agent
    policy_network:
        asdf
    target_network:
        asdf
    """

    def __init__(self, input_shape, n_actions, optimizer, gamma=0.99, C=4,
                 min_eps=0.01, max_eps=1, device='GPU', checkpoint_path='scores'):
        super(DuelingDDQNAgent, self).__init__(input_shape, n_actions, optimizer, gamma, C,
                                               min_eps, max_eps, device, checkpoint_path)

        self.memory = ReplayMemory(input_shape)
        self.policy_network = DuelingDDQN(input_shape, n_actions)
        self.target_network = DuelingDDQN(input_shape, n_actions)

    def choose_action(self, state):
        return 0

    def calculate_loss(self):
        return 0

    def back_propagate(self):
        return 0
