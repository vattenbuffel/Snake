import numpy as np
# import npc
from collections import deque
import random

import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


class Agent:
    def __init__(self, npc):
        self.discount = 0.9
        self.training_size = 2500  # Minimum number of steps in a memory to start training
        self.update_target_every = 10  # Terminal states (end of episodes)
        self.target_update_counter = 0
        self.train_counter_max = 250
        self.train_counter = self.train_counter_max
        self.memory = deque(maxlen=50000)  # Should be in the form [old_state, new_state, action, reward, if_terminal_state]
        self.verbose = 0

        self.n_outputs = npc.n_actions
        self.n_inputs = len(npc.old_state)

        self.epsilon_max = 0.999
        self.epsilon = self.epsilon_max
        self.epsilon_decay = 0.0005#0.9999
        self.epsilon_min = 0.0001

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self):
        model = Sequential()

        model.add(Dense(256, activation='linear', input_dim=self.n_inputs))
        model.add(Dense(256, activation='linear'))
        model.add(Dense(self.n_outputs))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['mae'])
        #model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def update(self, npc):
        self.train_counter -= 1
        self.update_memory(npc)
        self.train()
        self.get_action(npc)

    def update_memory(self, npc):
        #print('q-learning.old_state', npc.old_state)
        #print('q-learning.new_state', npc.new_state)
        #print('q-learning.reward', npc.reward)
        #print()
        self.memory.append(np.array([npc.old_state, npc.new_state, npc.action, npc.reward, npc.terminal_state]))

    def train(self, force=False):
        if not force and self.train_counter > 0:
            return

        self.train_counter = self.train_counter_max

        try:
            train_batch = random.sample(self.memory, self.training_size)
        except:
            train_batch = self.memory

        old_states = np.array([x[0] for x in train_batch])
        old_qs_list = self.model.predict(old_states)

        new_states = np.array([x[1] for x in train_batch])
        new_qs_list = self.target_model.predict(new_states)

        x = []
        y = []

        for i, (old_state, new_state, action, reward, in_terminal_state) in enumerate(train_batch):
            if not in_terminal_state:
                max_new_q = np.max(new_qs_list[i])
                updated_q = reward + self.discount * max_new_q
            else:
                updated_q = reward

            old_qs = old_qs_list[i]
            old_qs[action] = updated_q

            x.append(old_state)
            y.append(old_qs)

        self.model.fit(np.array(x), np.array(y), epochs=150, batch_size=self.training_size, verbose=self.verbose, shuffle=False)

        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_action(self, npc):
        # Sometime return a random action
        self.epsilon -= self.epsilon_decay
        #if self.epsilon < self.epsilon_min:
        #    self.epsilon = self.epsilon_max

        if self.epsilon < 0:
            print('epsilon < 0')
            self.epsilon = self.epsilon_max
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_outputs)

        state = np.array([npc.new_state])
        return np.argmax(self.model.predict(state))
