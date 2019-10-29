import gym
import numpy as np
import DeepQNetworkV2
from tqdm import tqdm
from keras.utils.np_utils import to_categorical   
import pickle as pkl

EPISODES = 30_000

EPSILON = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 50
RENDER = True

# make new environment
env = gym.make('MountainCar-v0')

# init state_size and action_space
norm = np.subtract(env.observation_space.high, env.observation_space.low)
state_size = env.observation_space.shape
action_size = env.action_space.n

# get space low and high
low = env.observation_space.low
high = env.observation_space.high

# because space is not discrete, make it discrete with s points
s = 200
position_range = np.linspace(low[0], high[0], s)
velocity_range = np.linspace(low[1], high[1], int(s/5))

dqn = DeepQNetworkV2.DQN(state_size, action_size)
episode_rewards = []

QTable = np.load('qtable2.npy')

    

for episode in tqdm(range(1, EPISODES+1), unit="episode"):
    dqn.tensorboard.step = episode

    step = 1
    current_state = np.add(np.divide(env.reset(), norm), env.observation_space.low)
    done = False
    episode_reward = 0

    while not done and step <= 200:
        if np.random.random() > EPSILON:
            action = np.argmax(dqn.get_qs(current_state))
        else:
            action = np.random.randint(0, action_size)
        
        new_state, reward, done, info = env.step(action)
        new_state = np.add(np.divide(new_state, norm), env.observation_space.low)
        episode_reward += reward

        if RENDER and not episode % AGGREGATE_STATS_EVERY:
            env.render()
        
        dqn.update_replay_memory((current_state, action, reward, new_state, done))
        dqn.train(done, step)

        current_state = new_state
        step += 1

    episode_rewards.append(episode_reward)
    
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY
        EPSILON = max(EPSILON, MIN_EPSILON)

    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(episode_rewards[-AGGREGATE_STATS_EVERY:])/len(episode_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(episode_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(episode_rewards[-AGGREGATE_STATS_EVERY:])
        dqn.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward)
        
        # dqn.save_model(max_reward, min_reward, average_reward)
