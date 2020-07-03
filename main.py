import gym
from gym import wrappers
import numpy as np
import time
import os
from gym.wrappers import AtariPreprocessing, FrameStack
from agents.DuelingDDQNAgent import DuelingDDQNAgent
from utils import save_agent
import torch



def colab():
    root_path = '/content/drive/My Drive/Projekti/ORI_RL'
    best_score = -np.inf
    # env = gym.make('SpaceInvadersNoFrameskip-v4')
    env = gym.make('BreakoutNoFrameskip-v4')
    env = AtariPreprocessing(env)
    env = FrameStack(env, num_stack=4)

    agent = DuelingDDQNAgent(input_shape=(env.observation_space.shape), n_actions=env.action_space.n, cutoff=1e6,
                             batch_size=32, C=10000)
    epochs = 10000
    game_steps = 0
    total_steps = 0
    scores, loss_plot = [], []

    for i in range(epochs):
        done = False
        observation = env.reset()

        score = 0
        start_time = time.time()
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward

            agent.memory.push(observation, action, next_observation, reward, int(done))
            loss = agent.calculate_loss_and_backprop()
            if loss > 0:
                loss_plot.append(loss)
            observation = next_observation

            game_steps += 1
            total_steps += 1

        scores.append(score)

        if score > best_score:
            best_score = score
            save_agent(agent, os.path.join(root_path, 'results', env.game), 'best_model')

        print('Game {} Score {} Best Score {} ε {:.2f} Game Steps {} Total Steps {}'.format(
            i, score, best_score, agent.eps, game_steps, total_steps
        ))
        print('Time taken for game {} sec\n'.format(time.time() - start_time))

        if i % 50 == 0:
            print("------------------------------")
            print('Last 50 games average {}'.format(sum(scores[-50:]) / len(scores[-50:])))
            print("------------------------------\n")
            save_agent(agent, os.path.join(root_path, 'results', env.game), 'latest_model')

        game_steps = 0

    save_agent(agent, os.path.join(root_path, 'results', env.game), 'final_model')


if __name__ == '__main__':
    colab()
