"""
    @Description: This file contains some functions for saving plots and animations.
    @Author     : Erfan Fathi
    @Date       : 26 May 2023
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import gym
from q_table import discretize_state

def save_frames_as_gif(frames, path="./", filename="gym_animation.gif"):
    """
        Description       : This function saves the frames as a gif file.
        Args:
            frames       : The frames of the animation.
            path         : The path for saving the gif file.
            filename     : The name of the gif file.
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def plot_reward(reward_list, num_episodes, average_magnitude=100, path="./", file_name="average_reward.png"):
    """
        Description       : This function plots the average reward.
        Args:
            reward_list  : The list of rewards.
            num_episodes : The number of episodes.
            average_magnitude : The magnitude of averaging.
            path         : The path for saving the plot.
            filename     : The name of the plot.
    """
    avg_reward = np.zeros(num_episodes//average_magnitude)
    for i in range(num_episodes//average_magnitude):
        avg_reward[i] = np.mean(reward_list[i*average_magnitude:(i+1)*average_magnitude])
    plt.plot(np.arange(0, num_episodes, average_magnitude), avg_reward)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.savefig(path + file_name)

def render_and_save_frames(learner, bins, num_steps, num_episodes, path="./", file_name="cartpole.gif"):
    """
        Description       : This function renders the frames and saves them as a gif file.
        Args:
            learner      : The learner agent.
            bins         : The bins for discretizing the state space.
            num_steps    : The number of steps.
            num_episodes : The number of episodes.
            path         : The path for saving the gif file.
            filename     : The name of the gif file.
    """
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    frames = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        state_discrete = discretize_state(state, bins)
        for _ in range(num_steps):
            frames.append(env.render())
            action = learner.policy.get_action(state_discrete)
            state, _, done = env.step(action)[:3]
            state_discrete = discretize_state(state, bins)
            if done:
                break
    save_frames_as_gif(frames, path=path, filename=file_name)
    env.close()

