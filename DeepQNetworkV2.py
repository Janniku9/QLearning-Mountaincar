from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
import ModifiedTensorBoard
from collections import deque
import numpy as np
import random as random
import time

REPLAY_MEM_SIZE = 50_000
MIN_REPLAY_MEM_SIZE = 10_000
MODEL_NAME = "256x2xCNN"
BATCH_SIZE = 32
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5



class DQN:
    def __init__ (self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space=action_space

        self.model = self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEM_SIZE)

        self.tensorboard = ModifiedTensorBoard.MTB(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        self.target_update_counter = 0

    def create_model (self):

        model = Sequential()

        model.add(Dense(64,input_shape=self.observation_space))
        model.add(Activation("relu"))
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dense(self.action_space, activation="relu"))
        model.compile(
            loss="mse", 
            optimizer=Adam(lr=0.001), 
                metrics=['accuracy'])
        return model

    def update_replay_memory (self, transition):
        self.replay_memory.append(transition)

    def get_qs (self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0] # normalize

    def train(self, terminnal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEM_SIZE:
            return
        
        batch = random.sample(self.replay_memory, BATCH_SIZE)
        current_states = np.array([transition[0] for transition in batch]) # normalize
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in batch]) # normalize
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(batch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
        
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            Y.append(current_qs)
        
        self.model.fit(np.array(X), np.array(Y), batch_size=BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminnal_state else None)

        if terminnal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def save_model(self, max_reward, min_reward, average_reward):
        self.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
    