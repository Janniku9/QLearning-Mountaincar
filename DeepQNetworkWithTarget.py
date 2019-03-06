import Memory

import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DeepQNetworkWithTarget():

    def __init__ (self, state_size, action_size, max_memory, discount, epsilon, epsilom_min, epsilon_decay, tau, learning_rate):
        self.state_size = state_size
        self.action_size = action_size

        # parameter
        self.mem = Memory.Memory(max_memory)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilom_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau

        # build model
        self.model = self.build_model()
        self.target = self.build_model()

    def build_model(self):
        # init new model
        model = Sequential()

        # add layers
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        # compile network
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def store_sample(self, state, action, reward, next_state, done):
        self.mem.add_sample(state, action, reward, next_state, done)

    def choose_next_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def replay(self, batch_size):
        # get batch
        batch = self.mem.get_batch(batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward

            if not done:
                target = (reward + self.discount * np.amax(self.target.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        weights = self.model.get_weights()
        target_weights = self.target.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1-self.tau)
        self.target.set_weights(target_weights)
