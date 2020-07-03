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
                 batch_size=32, min_eps=0.01, max_eps=1, cutoff=1e6, device='GPU'):
        super(DuelingDDQNAgent, self).__init__(input_shape, n_actions, gamma, batch_size,
                                               min_eps, max_eps, cutoff, device)

        self.memory = ReplayMemory(input_shape)
        self.policy_network = DuelingDDQN(input_shape, n_actions, self.device)
        self.target_network = DuelingDDQN(input_shape, n_actions, self.device)

        self.optimizer = getattr(torch.optim, optimizer)(self.policy_network.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.C = C
        self.C_counter = 0

    def choose_action(self, state):
        # should the agent explore
        if torch.rand(1) < self.eps:
            return np.random.choice(self.possible_actions)

        # else exploit
        with torch.no_grad():
            state = np.array([state], copy=False, dtype=np.float16)
            tensor = torch.from_numpy(state).float().to(self.device)
            return torch.argmax(self.policy_network(tensor)).item()

    def calculate_loss_and_backprop(self):
        if self.memory.counter < self.batch_size:
            return -1

        states, actions, next_states, rewards, terminals = self.memory.sample(self.batch_size)
        # states = states.float().to(self.device)
        # next_states = next_states.float().to(self.device)
        # rewards = rewards.float().to(self.device)
        ###########
        states = torch.from_numpy(states).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        ############

        all_idx = torch.arange(self.batch_size)
        # TODO SKONTAJ STO OVDE BACA GRESKU ZA ACTIONS INDEKSE
        # print(indices, actions)
        policy_out = self.policy_network(states)[all_idx, actions]
        # disable automatic gradient calc
        with torch.no_grad():
            policy_next_out = self.policy_network(next_states)
            target_out = self.target_network(next_states)
        ####
        # policy_next_out = self.policy_network.forward(next_states)
        # target_out = self.target_network.forward(next_states)
        ####

        # find best action for each next state
        max_actions = torch.argmax(policy_next_out, dim=1)

        # set all terminal states vals to zero so
        # that only rewards is present
        target_out[terminals] = 0.0

        y = rewards + self.gamma * target_out[all_idx, max_actions]
        self.optimizer.zero_grad()
        loss = self.criterion(y, policy_out)
        loss.backward()
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

        self.update_epsilon()


if __name__ == '__main__':
    x = DuelingDDQNAgent([32, 32, 32], 5, optimizer='SGD')

    for i in range(int(1e6) +200):
        x.update_epsilon()
    print('succcces')
