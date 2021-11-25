#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 01:19:56 2021

@author: yizhang
"""

from simple_playgrounds.playground.layouts import SingleRoom
from simple_playgrounds.agent.controllers import External
from simple_playgrounds.agent.agents import BaseAgent
from simple_playgrounds.device.sensors import  Lidar, Touch
import numpy as np
from numpy import pi
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import TQC
from dynamic_obstacle_avoidance import myEnv

    
# It evaluates the performance of the model. 
# It takes a model as input, output the mean #steps per episode and mean #collisions per episode

def evaluate(model, num_episode = 50):
    num_steps = []
    num_collisions = []
    for i in range(num_episode):
        obs = env.reset()
        n_steps = 200
        collision = 0
        for step in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            dist = obs[0][1]
            if reward + dist == -0.5:
                collision += 1
            env.render(mode='console')
            if done:
                print("Done!", "reward=", reward)
                break
        num_steps.append(step)
        num_collisions.append(collision)
    return np.mean(num_steps), np.mean(num_collisions)


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


# Instantiate the env
env = myEnv(my_playground)
# wrap it
env = make_vec_env(lambda: env, n_envs=1)


# Load a model 
model = SAC.load("models/sac")
# model = TQC.load("models/tqc")
# model = PPO.load("models/ppo")
# model = A2C.load("models/a2c")
print ("Model loaded successfully")


steps_SAC, collisions_SAC = evaluate(model, 50)
print ("SAC mean steps per episode",steps_SAC)
print("SAC mean collisions per episode", collisions_SAC)