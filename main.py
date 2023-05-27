"""
    @Description: This file is the main file for running the q-learning and sarsa algorithms on the cartpole-v1 environment.
    @Author     : Erfan Fathi
    @Date       : 26 May 2023
"""

import argparse
import gym
import numpy as np
from q_table import Qtable
from q_agent import QLearner
from sarsa_agent import SarasLearner
from utils import plot_reward, render_and_save_frames

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='q_learning', help='The algorithm to use. It can be either q_learning or sarsa.')
parser.add_argument('--alpha', type=float, default=0.1, help='The learning rate.')
parser.add_argument('--gamma', type=float, default=0.995, help='The discount factor.')
parser.add_argument('--epsilon', type=float, default=0.1, help='The exploration rate.')
parser.add_argument('--num_episodes', type=int, default=1000, help='The number of episodes.')
parser.add_argument('--num_steps', type=int, default=500, help='The number of steps.')
parser.add_argument('--num_bins', type=int, default=100, help='The number of bins for discretizing the state space.')
parser.add_argument('--seed', type=int, default=100, help='The seed for the random number generator.')
args = parser.parse_args()

# create string for the file name
file_name = args.algorithm + "_alpha_" + str(args.alpha) +\
            "_gamma_" + str(args.gamma) + "_epsilon_" +\
            str(args.epsilon) + "_num_episodes_" +\
            str(args.num_episodes) + "_num_steps_" +\
            str(args.num_steps) + "_num_bins_" + str(args.num_bins)

# print the file name
print("The file name is: ", file_name)

# define the environment
env = gym.make('CartPole-v1')
np.random.seed(args.seed)

if args.algorithm == "q_learning":
    # define the q-table
    q_table, bins = Qtable(env.observation_space, env.action_space, args.num_bins)

    # instantiate the q-learner agent
    q_learner = QLearner(args.alpha, args.gamma, args.epsilon, q_table, bins, env, args.seed)
    # learn the q-table
    reward_list = q_learner.learn(args.num_episodes, args.num_steps)

    # plot
    plot_reward(reward_list, args.num_episodes, 100, "./plots/", file_name + ".png")

    # render and save the frames
    render_and_save_frames(q_learner, bins, num_steps=100, num_episodes=10, path="./videos/", file_name=file_name + ".gif")
elif args.algorithm == "sarsa":
    # define the q-table
    q_table, bins = Qtable(env.observation_space, env.action_space, args.num_bins)

    # instantiate the q-learner agent
    sarsa_learner = SarasLearner(args.alpha, args.gamma, args.epsilon, q_table, bins, env, args.seed)
    # learn the q-table
    reward_list = sarsa_learner.learn(args.num_episodes, args.num_steps)

    # plot
    plot_reward(reward_list, args.num_episodes, 100, "./plots/", file_name + ".png")

    # render and save the frames
    render_and_save_frames(sarsa_learner, bins, num_steps=100, num_episodes=10, path="./videos/", file_name=file_name + ".gif")
else:
    raise ValueError("The algorithm should be either q_learning or sarsa.")