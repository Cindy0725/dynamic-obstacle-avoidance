#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 01:12:04 2021

@author: yizhang
"""
import numpy as np
import matplotlib.pyplot as plt


def plt_image(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()

# Convert the sensor values from 2d array to 1d
def array2d_to_1d(array_2d):
    array1d = np.zeros(len(array_2d))
    for i in range(len(array_2d)):
        if len(array_2d[i]) == 1:
            array1d[i] = array_2d[i][0] 
    return array1d


# The following functions are used to generate actions for the dynamic obstacle. 
# It takes the obstacle as input, output the actions of the obstacle (dictionary of {(actuator, value)}).

# Trajectory 1 of the obstacle
def generate_obstacle(agent):
    actions = {}
    for actuator in agent.controller.controlled_actuators:
        actions[actuator] = 0
    actions[agent.controller.controlled_actuators[1]] = 0.3
    actions[agent.controller.controlled_actuators[0]] = 0.5
    return actions

# Trajectory 2 of the obstacle
def generate_obstacle_2(agent, last_step):
    actions = {}
    l_step = 0
    for actuator in agent.controller.controlled_actuators:
        actions[actuator] = 0
    actions[agent.controller.controlled_actuators[0]] = 0.2
    if agent.coordinates[0][1] <= 60:
        l_step = 0.2
    if agent.coordinates[0][1] >= 140: 
        l_step = -0.2
    elif 55 < agent.coordinates[0][1] < 145:
        l_step = last_step
    actions[agent.controller.controlled_actuators[0]] = l_step
    return actions, l_step


# Calculate the distance from our agent to the goal
def dist_to_destination(agent, goal):
    x = agent.coordinates[0][0]
    y = agent.coordinates[0][1]
    dist =((x-goal[0])**2 + (y-goal[1])**2)**(1/2)
    t = ((goal[0])**2 + (goal[1])**2)**(1/2)
    norm_dist = dist / t
    return norm_dist


# Check if the agent is colliding with the obstacle or the wall.
def check_collision(sensor_values, agent):
    collision = False
    max_val = max(sensor_values)
    indices = []
    for i in range(len(sensor_values)):
         if sensor_values[i] == max_val:
                indices.append(i)
    in_front = True
    for ind in indices:
        if 13 <= ind < 27:
            in_front = True
            break

    if max_val > 0.9 and in_front:
        collision = True
    return collision

# Currently we don't use this function
def close_collision(sensor_values, agent):
    collision = False
    max_val = max(sensor_values)
    indices = []
    for i in range(len(sensor_values)):
         if sensor_values[i] == max_val:
                indices.append(i)
    in_front = False
    for ind in indices:
        if 13 <= ind < 27:
            in_front = True
            break

    if max_val > 0.7 and in_front:
        collision = True
    return collision


# Convert the action values from the reinforcement learning algorithms to dictionary of {(actuator, value)}
def custom_actions(action_values, agent):
    actions = {}
    longitudinal_force = action_values[0]
    angular_velocity = action_values[1]
    lateral_force = action_values[2]
    actions[agent.longitudinal_force] = longitudinal_force
    actions[agent.rotation_velocity] = angular_velocity
    actions[agent.lateral_force] = lateral_force
    actions[agent.activate] = 0
    actions[agent.grasp] = 0
    return actions

# Calculate the angle from our agent to the goal
def theta_to_destination(agent, goal):
    x = agent.coordinates[0][0]
    y = agent.coordinates[0][1]
    total = np.arctan((goal[0]-y)/(goal[1]-x))
    theta = np.radians(total) - agent.coordinates[1]
    theta_norm = theta / (2*np.pi)
    return theta_norm


# The following functions are used to generate actions for the moving goal. 
# It takes the goal agent as input, output the actions of the goal (dictionary of {(actuator, value)}).
def goal_moving(agent, last_step):
    actions = {}
    l_step = 0
    for actuator in agent.controller.controlled_actuators:
        actions[actuator] = 0
    actions[agent.controller.controlled_actuators[0]] = 0.2
    if agent.coordinates[0][0] <= 100:
        l_step = -0.2
    if agent.coordinates[0][0] >= 250: 
        l_step = 0.2
    elif 95 < agent.coordinates[0][0] < 255:
        l_step = last_step
    actions[agent.controller.controlled_actuators[0]] = l_step
    return actions, l_step

def goal_moving_2(agent, last_step):
    actions = {}
    l_step = 0
    for actuator in agent.controller.controlled_actuators:
        actions[actuator] = 0
    actions[agent.controller.controlled_actuators[0]] = 0.2
    if agent.coordinates[0][1] <= 60:
        l_step = 0.2
    if agent.coordinates[0][1] >= 140: 
        l_step = -0.2
    elif 55 < agent.coordinates[0][1] < 145:
        l_step = last_step
    actions[agent.controller.controlled_actuators[0]] = l_step
    return actions, l_step