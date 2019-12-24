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
        # todo: remove unnccessary variables
        self.discount = 0.9
        self.training_size = 2500  # Minimum number of steps in a memory to start training
        self.update_target_every = 10  # Terminal states (end of episodes)
        self.target_update_counter = 0
        self.train_counter_max = 250
        self.train_counter = self.train_counter_max
        self.memory = [[[], []]]
        # self.memory = deque(maxlen=50000)  # Should be in the form [old_state, new_state, action, reward, if_terminal_state]
        self.verbose = 0

        self.n_outputs = npc.n_actions
        self.n_inputs = len(npc.old_state)

        self.epsilon_max = 0.999
        self.epsilon = self.epsilon_max
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.0001

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self):
        model = Sequential()

        model.add(Dense(256, activation='relu', input_dim=self.n_inputs))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.n_outputs))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['mae'])
        return model

    def update(self, npc):
        self.update_memory(npc)
        self.train()
        self.get_action(npc)

    def update_memory(self, npc):
        for i in range(len(self.memory)):
            if npc.old_state.tolist() not in self.memory[i][0]:
                self.memory[i].append(
                    np.array([npc.old_state, npc.new_state, npc.action, npc.reward, npc.terminal_state]))
                self.memory[i][0].append(npc.old_state.tolist())
                self.memory[i][1].append(npc.action)
                return

            elif npc.old_state.tolist() in self.memory[i][0] and npc.action == self.memory[i][1][
                self.memory[i][0].index(npc.old_state.tolist())]:
                return

        # If not yet added to any memory list start a new memory list
        new_memory_list = [[npc.old_state.tolist()], [npc.action],
                           np.array([npc.old_state, npc.new_state, npc.action, npc.reward, npc.terminal_state])]
        self.memory.append(new_memory_list)

    def train(self):
        index = np.random.randint(0, len(self.memory))
        train_batch = self.memory[index][2:len(self.memory[index]) + 1]

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

        self.model.fit(np.array(x), np.array(y), epochs=10, batch_size=self.training_size, verbose=self.verbose,
                       shuffle=False)

        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_action(self, npc):
        # Sometime return a random action
        self.epsilon *= self.epsilon_decay
        # if self.epsilon < self.epsilon_min:
        #    self.epsilon = self.epsilon_max

        if self.epsilon < self.epsilon_min:
            print('epsilon <', self.epsilon_min)
            self.epsilon = self.epsilon_max
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_outputs)

        state = np.array([npc.new_state])
        return np.argmax(self.model.predict(state))
