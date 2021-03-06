# Table of Content

- [Introduction](#introduction)
- [Simple-Playgrounds](#simple-playgrounds)
- [Elements of the environment](#elements-of-the-environment)
- [Training with Deep RL algorithms](#training-with-deep-rl-algorithms)
- [Evaluation & Comparison](#evaluation-and-comparison)
- [Programming](#programming)

# Introduction 
The objective of this project is to train a robot to avoid dynamic obstacles and arrive at a moving goal in a simulation environment. We train the robot with four deep reinforcement learning algorithms in stable-baselines 3: PPO, A2C, SAC, TQC, and compare their performance on avoiding moving obstacles. 

# Simple-playgrounds
Our dynamic obstacle avoidance environment is implemented based on *Simple-Playgrounds* (SPG) software library: https://github.com/mgarciaortiz/simple-playgrounds.  *Simple-Playgrounds* (SPG) is an easy-to-use, fast and flexible simulation environment for research in Deep Reinforcement Learning. It bridges the gap between simple and efficient grid environments, and complex and challenging 3D environments.  
The playgrounds are 2D environments where agents can move around and interact with scene elements. The game engine, based on Pymunk and Pygame, deals with simple physics, such as collision and friction. Agents can act through continuous movements and discrete interactive actions. They perceive the scene with realistic first-person view sensors, top-down view sensors, and semantic sensors.

# Elements of the environment
## Playground
Our agent move and perceive in a playground. A playground is described using a Cartesian coordinate system. Each element in the playground has a position (x,y, &theta), with x along the horizontal axis, y along the vertical axis, and theta the orientation, aligned on the horizontal axis. For example, the top left corner has position (0, 0).  
A playground has a size (width, length), with the width along x-axis, and length along y-axis.  
We created a playground with size (200, 300). 

## Agent, Obstacle and Goal
### Agent
In *Simple-Playgrounds*, our agent is a Base agent with a controller which can manage the actuators of the agent. We use an External controller to set actions of the agent from outside of the simulators (using reinforcemnt learning algorithms). 
Our agent perceive the surroudings with two sensors:
- Lidar Sensor  
The lidar sensor has 64 number of rays and 180 degrees field of view. Our agent use it to detect obstacles. 
- Touch Sensor  
The touch sensor has 36 number of rays and 360 degrees field of view. It is used to check whether our agent collides with the moving obstacle or the wall of our playground.  

Our agent starts from position (50, 50). 
### Obstacle
For each episode, our obstacle starts from a random position on the straight line between (110, 55) and (110, 180). Then it moves around following trajectory 1 or trajectory 2. 
Trajectory 1: It repeatedly moves in a straight line between position (110, 55) and position (110, 145).   
Trajectory 2: It starts from position (110, 60) and repeatedly moves along an elliptical trajectory. 
### Goal
Our goal can also move in the playground. It repeatedly moves along a straight line between position (220, 55) and position (220, 180). For each episode, our goal also starts from a random position on the straight line between (220, 55) and (220, 180).
## Action Space
Our action space is continuous and it contains 3 variables:
- Longitudinal force  
The longitudinal force is a float number in range [-1, 1]. The agent moves forward when it is positive and backward when negative. However, since the field of view of our lidar sensor is only 180 degrees, we don't want our agent move backward without perceiving the environment behind it. So we limit the range of longitudinal force to [0, 1].    
- Rotation velocity       
The rotation velocity is a float number in range [-1, 1]. The agent turns to right for a certain angle if it is positive and left if negative. 
- Lateral force   
The lateral force is a float number in range [-1, 1].

## State Space
Our state space is continuous with 66 variables:
- theta   
theta is the angle between our agent and the goal. It is normalized to [-1, 1].
- dist    
dist is the distance between our agent and the goal. It is normalized to [0, 1]. It is used to check whether our agent arrives at the goal.
- 64 lidar sensor values      
The sensor values are normalized to [0, 1].

## Reward
- If the agent arrives at the goal: +1 
- If the agent collides with the wall or moving obstacle: -0.5 

Reward shaping: we also minus the distance between our agent and the goal from the reward (reward -= dist).   
The episode will terminate if the agent arrives at the goal.

# Training with Deep RL algorithms
We train our agent with 4 deep reinforcement learning algorithms from *Stable-Baselines3*: https://github.com/DLR-RM/stable-baselines3. The algorithms are: PPO, A2C, SAC, TQC. 
We train 100000 timesteps for each algorithm. 

# Evaluation and Comparison
## Learning Speed
When training the algorithms, we check the mean reward per episode every 1000 steps. The smoothed learning curves of these algorithms are as following:

![image](https://github.com/Cindy0725/dynamic-obstacle-avoidance/blob/main/imgs/learning_curve_comparison.png)

According to the figures, the mean reward per episode keeps increasing in general for PPO, SAC and TQC, and TQC learns fastest.

![image](https://github.com/Cindy0725/dynamic-obstacle-avoidance/blob/main/imgs/A2C_2.png)

However, A2C can't converge to a satifying reward after 100000 timesteps. Its mean reward per episode even decreases as the number of timesteps increases.

## Performance
There are two criteria to assess the performance of the reinforcement learning algorithms: 
- Number of steps used to reach the goal per episode 
- Number of collisions per episode  

We run 50 episodes for each algorithm. Since the agent is expected to arrive at the goal within 100 steps, we set the maximum timestep to 100 for each episode.

****
	
|Algorithm|Mean #steps per episode|Mean #collisions per episode|
|---|---|---
|PPO|58.82|0.14
|A2C|>>100|2.48
|SAC|30.66|0.02
|TQC|35.04|0.00

****

From the table, we can see that SAC and TQC performs well on both avoiding dynamic obstacles and reaching a moving goal. However, A2C can not arrive at the goal since it almost stays in the same position as time goes. PPO learns slower and performs worse than SAC and TQC.

# Programming 
### directory *models*
In *models* directory, there are four models trained by SAC, TQC, PPO and A2C (timesteps = 100000). Their performances are shown in the Evaluation & Comparison section.
### file *solution.py* (testing)
*solution.py* create an environment with one agent, one moving obstacle and one moving goal. It loads model from *models directory*. Now running *solution.py* will display how the agent moves after training by TQC. It will also output the mean number of steps per episode and the mean number of collisions per episode in the console. If you want to see other models' performance, just modify the following line at the end of the file:

```python
model = TQC.load("models/tqc")
```

### file *train.py* (training)
*train.py* is used to train different models. Running *train.py* will train a model by SAC algorithm and save it to *models directory*. If you want to train models by PPO, TQC or A2C, just modify the following lines at the end of the file:

```python
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000, log_interval=4)
model.save("models/sac")
```

Please also remember to comment the following lines in *dynamic_obstacle_avoidance.py* since displaying the agent's performance while training is very time consuming. 

```python
plt_image(eng.generate_playground_image())		       #line 66
plt_image(eng.generate_agent_image(eng.agents[0]))	       #line 67

cv2.imshow('agent', eng.generate_agent_image(eng.agents[0]))   #line 97
cv2.waitKey(20)						       #line 98
```
