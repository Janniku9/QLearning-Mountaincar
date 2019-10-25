import numpy as np 
from numpy import random
import gym
import matplotlib.pyplot as plt

# make new environment
env = gym.make("MountainCar-v0")

# number of episodes
episodes = 60000

# learning rate a and exploration/exploitation rate e and discount rate d
a = 0.2
e = 1.0
d = 0.9
rewards = []
success_rate = []
success = 0.0

# get space low and high
low = env.observation_space.low
high = env.observation_space.high

# because space is not discrete i make it discrete with s points
s = 200
position_range = np.linspace(low[0], high[0], s)
velocity_range = np.linspace(low[1], high[1], int(s/5))

# get actions
actions = env.action_space.n

# init QTable with dimension  s x s/5 x actions
QTable = np.zeros((s, int(s/5), actions))

for eps in range(episodes):
    # reset episode and get initial state
    env.reset()
    old_position = np.searchsorted(position_range, 0)
    old_velocity = np.searchsorted(velocity_range, 0)

    # print number of iterations every 1000 iterations
    if eps % 1000 is 0:
        print (eps)

    # init done and reward
    done = False
    total_reward = 0

    # update exploration/exploitation rate
    e = e - 1.0/episodes    

    # run episode
    while (not done):
        if eps % 1000 is 0:
            env.render()

        # calculate next step
        if random.random() < e:
            next_action = random.randint(0, actions)
        else:
            next_action = np.argmax(QTable[old_position, old_velocity])
        
        # do step
        step = env.step(next_action)

        # round position and velocity to nearest entry in Q table
        position = np.searchsorted(position_range, step[0][0])
        velocity = np.searchsorted(velocity_range, step[0][1])
        reward = step[1]
        done = step[2]

        total_reward += reward

        # check if car reached top
        if done and 0.47 <= position_range[position]:
            QTable[position, velocity, next_action] = reward
            success = success + 1.0
        
        # update QTable
        else:
            deltaQ = a * (reward + d*np.max(QTable[position, velocity]) - QTable[old_position, old_velocity, next_action])
            QTable[old_position, old_velocity, next_action] += deltaQ

        old_position = position
        old_velocity = velocity
    rewards.append(total_reward)
    success_rate.append(success/(eps+1))
    if eps % 100 is 0:
        success = 0

plt.plot(success_rate)
plt.show()

plt.plot(rewards)
plt.show()