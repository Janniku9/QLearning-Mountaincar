import numpy as np 
from numpy import random
import gym

# make new environment
env = gym.make("MountainCar-v0")

# number of episodes
episodes = 10000

# learning rate a and exploration/exploitation rate e and discount rate d
a = 0.2
e = 1.0
d = 0.9
count = 0

# get space low and high
low = env.observation_space.low
high = env.observation_space.high

# because space is not discrete i make it discrete with s points
s = 100
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

    # init done and loop iterator
    done = False

    # update exploration/exploitation rate
    e = e - 1.0/episodes    

    # run episode
    while (not done):
        # if eps >= episodes - 100 or eps < 5:
            # env.render()

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

        # check if car reached top
        if done and 0.47 <= position_range[position]:
            QTable[position, velocity, next_action] = reward
            if eps >= episodes - 1000:
                count = count + 1
        
        # update QTable
        else:
            deltaQ = a * (reward + d*np.max(QTable[position, velocity]) - QTable[old_position, old_velocity, next_action])
            QTable[old_position, old_velocity, next_action] += deltaQ

        old_position = position
        old_velocity = velocity

print("success: ", count)