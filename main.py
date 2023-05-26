"""
Description: We want to implement q-learning algorithm and sarsa algorithm with epsilon-greedy policy
                to solve the cartpole-v1 problem in gym.
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from q_table import Qtable, discretize_state
from policy import EpsilonGreedyPolicy
from q_agent import QLearner
from sarsa_agent import SarasLearner
from utils import plot_reward, render_and_save_frames
import argparse

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

print("The arguments are: ", args)

# define the environment
env = gym.make('CartPole-v1')
np.random.seed(args.seed)

# define the q-table
q_table, bins = Qtable(env.observation_space, env.action_space, args.num_bins)

# # instantiate the q-learner agent
q_learner = QLearner(args.alpha, args.gamma, args.epsilon, q_table, bins, env, args.seed)
# learn the q-table
reward_list = q_learner.learn(args.num_episodes, args.num_steps)

# plot
plot_reward(reward_list, args.num_episodes, 100, "./plots/", "average_reward.png")

# render and save the frames
render_and_save_frames(q_learner, bins, num_steps=100, num_episodes=10, path="./videos/", file_name="cartpole.gif")
