from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminal'))


class ReplayMemory:
    """ Memory of an DQN agent

    Parameters
    -----------
    max_capacity: int
        maximum number of stored memories
    input_shape: array-like
        nd-array with the shape of one state

    Attributes
    -----------
    curr_idx: int
        index of first available element in memory
        (after reaching max_capacity it goes back to 0
        and starts overwriting old memories)
    states: nd-array
        nd-array containing all states in memory
    actions: array-like
        array with all actions that were taken
    next_states: nd-array
        nd-array with new state s' after action a
    rewards: array-like
        array of rewards for taking action a in state s
    terminals: array-like
        array of elements signaling whether the episode has terminated
    """

    def __init__(self, max_capacity, input_shape):
        self.max_capacity = max_capacity
        self.curr_idx = 0
        self.states = np.zeros((self.max_capacity, *input_shape), dtype=np.float16)
        self.actions = np.zeros(self.max_capacity, dtype=np.int8)
        self.next_states = np.zeros((self.max_capacity, *input_shape), dtype=np.float16)
        self.rewards = np.zeros(self.max_capacity, dtype=np.float16)
        self.terminals = np.zeros(self.max_capacity, dtype=np.bool)

    def push(self, *args):
        """ Adds a new transition to memory

        Parameters
        -----------
        args : tuple
            transition tuple (state, action, next_state, reward, terminal)
        """
        # Unpack the transition tuple
        self.assign(Transition(*args))
        # Update current index
        self.curr_idx = (self.curr_idx + 1) % self.max_capacity

    def sample(self, n_transitions):
        """ Samples a number of random transitions from memory

        Parameters
        -----------
        n_transitions: int
            number of samples to return

        Returns
        -----------
        transitions: tuple of array-like objects
            tuples (state, action, next_state, reward, terminal)
            each tuple element of n_transitions length
        """
        max_idx = min(self.curr_idx, self.max_capacity)
        indices = np.random.choice(max_idx, n_transitions, replace=False)
        return self.states[indices], self.actions[indices], self.next_states[indices], self.rewards[indices], \
            self.terminals[indices]

    def assign(self, transition):
        """ Unpacks the transition to memory """

        self.states[self.curr_idx] = transition.state
        self.actions[self.curr_idx] = transition.action
        self.next_states[self.curr_idx] = transition.next_state
        self.rewards[self.curr_idx] = transition.reward
        self.terminals[self.curr_idx] = transition.terminal
