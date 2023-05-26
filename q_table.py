"""
    @Description: This file contains some useful functions for the project.
    @Author     : Erfan Fathi
    @Date       : 25 May 2023
"""

import numpy as np

def Qtable(state_space, action_space, bin_size=100):
    """
        Description       : This function defines the q-table for the cartpole-v1 environment.
        Args:
            state_space   : The state space of the environment.
            action_space  : The action space of the environment.
            bin_size      : The number of bins for discretizing the state space.
        Returns:
            q_table       : The q-table for the cartpole-v1 environment.
            bins          : The bins for discretizing the state space.
        Info:
        state_space Shape : (4,)
        state_space High  : [ 4.8   inf  0.42  inf]
        state_space Low   : [-4.8 - inf -0.42 -inf]
    """
    # define the bins
    bins = np.zeros((state_space.shape[0], bin_size))

    # initialize the bins
    for i in range(state_space.shape[0]):
        bins[i] = np.linspace(state_space.low[i], state_space.high[i], bin_size)

    # define the q-table
    q_table = np.zeros((bin_size, bin_size, bin_size, bin_size, action_space.n))

    return q_table, bins


def discretize_state(state_space, bins):
    """
        Description        : This function discretizes the state space.
        Args:
            state_space    : The state space of the environment.
            bins           : The bins for discretizing the state space.
        Returns:
            state_discrete : The discretized state space.
    """
    state_discrete = np.zeros(state_space.shape)

    for i in range(state_space.shape[0]):
        state_discrete[i] = np.digitize(state_space[i], bins[i])

    return state_discrete.astype(np.int32)

# Unit test
# def test():
#     import gym
#     env = gym.make('CartPole-v1')
#     # test the q-table
#     q_table, bins = Qtable(env.observation_space, env.action_space)
#     print(q_table.shape)

#     # test the discretize_state function
#     state, _ = env.reset()
#     state_discrete = discretize_state(state, bins)
#     print(state_discrete)

#     # sample step
#     action = env.action_space.sample()
#     next_state = None
#     for i in range(100):
#         next_state, reward, done = env.step(action)[:3]

#     state_next_discrete = discretize_state(next_state, bins)
#     print(state_next_discrete)