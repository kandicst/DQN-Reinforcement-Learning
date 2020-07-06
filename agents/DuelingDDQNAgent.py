from ReplayMemory import ReplayMemory
from agents.Agent import Agent
from networks.DuelingDDQN import DuelingDDQN
import torch
import numpy as np


class DuelingDDQNAgent(Agent):
    """ Dueling Double Deep Q-Network Agent
    Parameters
    -----------
    optimizer: basestring
        name of agent optimizer
    lr: float
        learning rate for optimizer
    C: int
        how many steps to take before replacing target network

    Attributes
    -----------
    memory: ReplayMemory
        memory of the agent
    policy_network: DuelingDDQN
        network used to select actions
    target_network: DuelingDDQN
        network used to evaluate actions
    optimizer: torch.optim
        torch optimizer object
    criterion: torch.nn.loss
        torch loss object
    C_counter: int
        how many steps since last target network swap
    """

    def __init__(self, input_shape, n_actions, optimizer='RMSprop', lr=1e-4, gamma=0.99, C=10000,
                 batch_size=32, min_eps=0.1, max_eps=1, cutoff=1e6, second_cuttof=2.5e6, final_eps=0.01,
                 device='GPU', clip=10):
        super().__init__(input_shape, n_actions, gamma, batch_size,
                                               min_eps, max_eps, cutoff, device)

        self.memory = ReplayMemory(input_shape)
        self.policy_network = DuelingDDQN(input_shape, n_actions, self.device)
        self.target_network = DuelingDDQN(input_shape, n_actions, self.device)

        self.optimizer = getattr(torch.optim, optimizer)(self.policy_network.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.second_cuttof = second_cuttof
        self.final_eps = final_eps

        self.C = C
        self.C_counter = 0
        self.clip = clip

    def choose_action(self, state):
        # should the agent explore
        if torch.rand(1) < self.eps:
            return np.random.choice(self.possible_actions)

        # else exploit
        with torch.no_grad():
            state = np.array([state], copy=False, dtype=np.float16)
            ten = torch.from_numpy(state).float().to(self.device)
            return torch.argmax(self.policy_network(ten)).item()

    def calculate_loss_and_backprop(self):
        if self.memory.counter < self.batch_size:
            return -1

        states, actions, next_states, rewards, terminals = self.memory.sample(self.batch_size)
        states = torch.from_numpy(states).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)

        all_idx = torch.arange(self.batch_size)
        policy_out = self.policy_network(states)[all_idx, actions]
        # disable automatic gradient calc
        with torch.no_grad():
            policy_next_out = self.policy_network(next_states)
            target_out = self.target_network(next_states)

        # find best action for each next state
        max_actions = torch.argmax(policy_next_out, dim=1)

        # set all terminal states vals to zero so
        # that only rewards is present
        target_out[terminals] = 0.0

        y = rewards + self.gamma * target_out[all_idx, max_actions]
        self.optimizer.zero_grad()
        loss = self.criterion(y, policy_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.policy_network.parameters(), self.clip)
        self.optimizer.step()

        # after it has finished just update all necessary values
        self.update_counters()
        return loss.item()

    def update_counters(self):
        # update counters and possibly swap target network
        self.step_counter += 1
        self.C_counter += 1
        if self.C_counter % self.C == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        if self.eps == self.final_eps: return

        if self.step_counter == self.second_cuttof:
          self.eps = self.final_eps
        else:
          self.update_epsilon()

