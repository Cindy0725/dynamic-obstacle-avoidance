#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 01:42:11 2021

@author: yizhang
"""

import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3 import PPO, A2C, SAC
from sb3_contrib import TQC
from dynamic_obstacle_avoidance import myEnv
from simple_playgrounds.playground.layouts import SingleRoom
from simple_playgrounds.agent.controllers import External
from simple_playgrounds.agent.agents import BaseAgent
from simple_playgrounds.device.sensors import  Lidar, Touch
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
              # Mean training reward over the last 100 episodes
                  mean_reward = np.mean(y[-100:])
                  if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              
            if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                  # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True



# Initialize a 300 * 200 room with simple-playground.
my_playground = SingleRoom(size=(300, 200))
#  Add one agent, one obstacle and one goal to the room.
initial_position = (50,50)
initial_goal_2 = (220, 60)
my_agent = BaseAgent(controller=External(), interactive = True, radius = 10, lateral = True)
goal = BaseAgent(controller=External(), interactive = False, radius = 10)
my_agent.add_sensor(Touch(my_agent.base_platform, normalize = True, invisible_elements=my_agent.parts + goal.parts))
obstacle = BaseAgent(controller=External(), interactive = False, radius = 10)
my_agent.add_sensor(Lidar(my_agent.base_platform, normalize = True, invisible_elements=my_agent.parts + goal.parts))
my_playground.add_agent(my_agent, (initial_position, 1.5*pi/4))
my_playground.add_agent(obstacle, ((110,60), pi/2))
my_playground.add_agent(goal, (initial_goal_2, pi/2))


# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = myEnv(my_playground)
# Logs will be saved in log_dir/monitor.csv
env = Monitor(env, log_dir)


# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
# Create RL model
model = SAC("MlpPolicy", env, verbose=1)
# Train the agent
model.learn(total_timesteps=10000, callback=callback, log_interval=4)

# Helper from the library
results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "SAC")

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig('SAC.png')


plot_results(log_dir, "SAC")