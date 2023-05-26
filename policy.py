"""
    @Description: In This file we implement epsilon greedy policy
    @Author     : Erfan Fathi
    @Date       : 25 May 2023
"""

import numpy as np

class EpsilonGreedyPolicy:
    """
        Description: This class implements the epsilon greedy policy.
        Args:
            epsilon  : The exploration rate.
            q_table  : The q-table for the cartpole-v1 environment.
            env      : The cartpole-v1 environment.
    """
    def __init__(self, epsilon, q_table, env):
        self.epsilon = epsilon
        self.q_table = q_table
        self.env = env

    def get_action(self, state):
        """
            Description: This function returns an action based on the epsilon greedy policy.
            Args:
                state      : The current state.
            Returns:
                action     : The action.
        """
        if np.random.rand() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state[0], state[1], state[2], state[3]])
        return action
    
class EpsilonDecayGreedyPolicy:
    pass

class UCBPolicy:
    pass