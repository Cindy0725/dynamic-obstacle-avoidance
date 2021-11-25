#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 01:07:35 2021

@author: yizhang
"""
import numpy as np
import gym
from gym import spaces
import random
from simple_playgrounds.engine import Engine
from helper_funcs import array2d_to_1d, generate_obstacle_2, dist_to_destination, check_collision, theta_to_destination, goal_moving_2, custom_actions, plt_image
import cv2



class myEnv(gym.Env):
#     destination = (170, 170)
    def __init__(self, my_playground):
#         super(myEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        
        self.action_space = spaces.Box(np.array([0, -1, -1]), np.array([1, 1, 1]), dtype=np.float32)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(np.array([-1]+[0]*65), np.array([1]*66), dtype=np.float32)
        self.engine = Engine(time_limit=100000, playground = my_playground)
        self.threshold = 0.1
        self.last_step = 0.2
        self.goal_initial = [220, 60]
        self.goal = [220, 60]
    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """

        eng = self.engine
        eng.reset()
        # For each episode we move the initial position of the obstacle
        n = random.randint(0,70)
        l = 0.2
        for i in range(n):
            actions = {}
            actions[eng.agents[0]] = custom_actions(np.array([0,0,0]), eng.agents[0])
            actions[eng.agents[1]], l = generate_obstacle_2(eng.agents[1], l)
            eng.step(actions)
            eng.update_observations()

        # For each episode we move the initial position of the goal
        m = random.randint(0,70)
        l = 0.2
        for i in range(m):
            actions = {}
            actions[eng.agents[0]] = custom_actions(np.array([0,0,0]), eng.agents[0])
            actions[eng.agents[2]], l = goal_moving_2(eng.agents[2], l)
            eng.step(actions)
            eng.update_observations()
        
        self.goal = []
        self.goal.append(eng.agents[2].coordinates[0][0])
        self.goal.append(eng.agents[2].coordinates[0][1])
        
        plt_image(eng.generate_playground_image())
        plt_image(eng.generate_agent_image(eng.agents[0]))

        self.engine = eng
        sensor_vals_lidar = eng.agents[0].sensors[1].sensor_values
        if sensor_vals_lidar is None:
            sensor_vals_lidar = np.array([0]*64)
        else: 
            sensor_vals_lidar = array2d_to_1d(sensor_vals_lidar)
        
        dist = dist_to_destination(eng.agents[0], self.goal)
        theta = theta_to_destination(eng.agents[0], self.goal)
        obs = np.insert(sensor_vals_lidar, 0, dist)
        obs = np.insert(obs, 0, theta)
        return obs.astype(np.float32)

    def step(self, action_values):
        
        # Here we custom the action_values, generate the "actions" that can be taken by our playground engine
        actions = {}
        eng = self.engine
        actions[eng.agents[0]] = custom_actions(action_values, eng.agents[0])
        actions[eng.agents[2]], l = goal_moving_2(eng.agents[2], self.last_step)
#         actions[eng.agents[1]] = generate_obstacle(eng.agents[1])
        actions[eng.agents[1]], l = generate_obstacle_2(eng.agents[1], self.last_step) 
        self.last_step = l
        
        # Apply the actions 
        eng.step(actions)
        eng.update_observations()

        cv2.imshow('agent', eng.generate_agent_image(eng.agents[0]))
        cv2.waitKey(20)
        
        self.engine = eng
        
        # Custom the sensor values and distance to destination after our agent takes action
        sensor_vals_lidar = eng.agents[0].sensors[1].sensor_values
        sensor_vals_lidar = array2d_to_1d(sensor_vals_lidar)
        sensor_vals_touch = eng.agents[0].sensors[0].sensor_values
        sensor_vals_touch = array2d_to_1d(sensor_vals_touch)
        
        self.goal = []
        self.goal.append(eng.agents[2].coordinates[0][0])
        self.goal.append(eng.agents[2].coordinates[0][1])
        dist = dist_to_destination(eng.agents[0], self.goal)
        theta = theta_to_destination(eng.agents[0], self.goal)
        obs = np.insert(sensor_vals_lidar, 0, dist)
        obs = np.insert(obs, 0, theta)

        
#         If our agent collides with the obstacle (agent2), then gives reward -0.5. 
#         If our agent arrives at the destination, then gives reward 1.
#         Also do reward shaping: reward -= dist. 
#         This doesn't work for A2C: hard exploration ---> sparse reward problem
        reward = 0
        if check_collision(sensor_vals_touch, eng.agents[0]):
            reward = -0.5
#             print ("collision with obstacle!!!")
        if dist < self.threshold:
            reward = 1
        reward += -dist


        # Are we at the terminate state?
        done = bool(dist < self.threshold)
        if dist < self.threshold:
            print ("Goal reached!")

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return obs.astype(np.float32), reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
          raise NotImplementedError()

    def close(self):
        pass
    