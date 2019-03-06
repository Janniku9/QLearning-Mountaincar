import gym
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

import Memory
import DeepQNetworkWithTarget

write = True
load = False

# make new environment
env = gym.make('CartPole-v0')

# init state_size and action_space
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# init new DeepQNetwork
dqn = DeepQNetworkWithTarget.DeepQNetworkWithTarget(state_size, action_size, 1000, 0.95, 1.0, 0.01, 0.995, 0.125, 0.01)
if load:
    dqn.load('weights.h5')

done=False
batch_size=32

episodes = 1000
steps = 200

rewards = []
success = 0

render = False
total_success = 0

for eps in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    total_reward = 0
    for s in range(steps):
        if render:
            env.render()
        elif eps%100 is 0:
            env.render()

        next_action = dqn.choose_next_action(state)
        next_state, reward, done, _ = env.step(next_action)

        total_reward += reward

        next_state = np.reshape(next_state, [1, state_size])

        dqn.store_sample(state, next_action, reward, next_state, done)
        state = next_state

        if done:
            # print(eps)
            if s is steps-1:
                success += 1
            break
    if len(dqn.mem.mem) > batch_size:
        dqn.replay(batch_size)
        dqn.update_target()
    rewards.append(total_reward)
    if eps%100 is 0:
            if write:
                dqn.save('weights.h5')
                print ("saved weights")
            print("succesful: ", success)
            success = 0
      
plt.plot(rewards)
plt.show()