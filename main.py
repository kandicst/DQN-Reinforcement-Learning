import gym
from gym import wrappers
import numpy as np
import time
from agents.DuelingDDQNAgent import DuelingDDQNAgent
from utils import plot_learning_curve, make_env

if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    load_checkpoint = False
    n_games = 5
    agent = DuelingDDQNAgent(input_shape=(env.observation_space.shape), n_actions=env.action_space.n)

    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    figure_file = 'plots/nesto.png'
    times=[]
    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        start_time = time.time()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                method_time = time.time()
                agent.memory.push(observation, action,observation_,
                                  reward, int(done))
                agent.calculate_loss_and_backprop()
                times.append(time.time() - method_time)
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        if score > best_score:
            best_score = score
        print("--- %s seconds ---" % (time.time() - start_time))
        print("--- Average call %s seconds ---" % (sum(times) / len(times) ))
        print('episode: ', i, 'score: ', score,
              ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
              'epsilon %.2f' % agent.eps, 'steps', n_steps)

        eps_history.append(agent.eps)
        if load_checkpoint and n_steps >= 18000:
            break

    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
