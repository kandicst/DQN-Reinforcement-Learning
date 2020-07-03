import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
import os


def save_agent(agent, path, name):
    if not os.path.isdir(path):
        os.mkdir(path)

    full_path = os.path.join(path, name)
    torch.save(agent.policy_network.state_dict(), full_path)


def load_agent(agent, path):
    agent.policy_network.load_state_dict(torch.load(path))