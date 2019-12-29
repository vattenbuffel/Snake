import numpy as np
import time
from os import walk
import random

import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import TensorBoard


# todo: try with a pure list memory
# todo: do an evaluation method, try all possibilites [1 0 0 0 0 1 0 0] for example
# todo: add drop out
# todo: train on every iteration of the for-loop?
class Agent:
    def __init__(self, npc):
        self.discount = 0.9
        self.training_batch_size = 2500
        self.update_target_every = 10
        self.target_update_counter = 0
        # self.memory = [[[], []]]
        self.memory = []
        self.verbose = 0

        self.n_outputs = npc.n_actions
        self.n_inputs = len(npc.old_state)

        self.epsilon_max = 0.999
        self.epsilon = self.epsilon_max
        self.epsilon_decay = 0.996
        self.epsilon_min = 0.001

        self.save_max = 500
        self.save_counter = 0

        self.training_mode = self.get_training_mode()
        self.model = self.get_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # Ensure random actions when training
        if self.training_mode:
            self.epsilon_max = 0.999
            self.epsilon = self.epsilon_max
            self.epsilon_decay = 0.999
            self.epsilon_min = 0.001

        # Ensure no random actions when training
        else:
            self.epsilon_max = 0
            self.epsilon = self.epsilon_max
            self.epsilon_decay = 0
            self.epsilon_min = 0

    # Ask the user if an already existing model should be used, a new be created and whether it should be trained or not
    def get_model(self):
        print('\nShould a new model be created or should a pre existing one be used? \n1) New model \n2) Pre existing '
              'model')
        ans = input('Choice: ')

        if ans == "1":
            print("A new model is created")
            return self.create_model()
        elif ans == "2":
            mypath = "./trained_agents/"
            files = []
            for (dirpath, dirnames, filenames) in walk(mypath):
                files.extend(filenames)
                break

            h5_files = [f if f.endswith(".h5") else False for f in files]
            while False in h5_files:
                h5_files.remove(False)

            print("\nWhat model should be loaded?")
            [print(h5_files.index(name) + 1, ") ", name) for name in h5_files]

            ans = input("Choice:")
            if not ans.isnumeric() or int(ans) > len(h5_files):
                print('Incorrect input')
                exit()

            model = load_model(mypath + h5_files[int(ans) - 1])
            if not model.get_input_shape_at(0) == (None, self.n_inputs):
                print("The input structure of the model doesn't comply with the code")
                exit()
            elif not model.get_output_shape_at(-1) == (None, self.n_outputs):
                print("The output structure of the model doesn't comply with the code")
                exit()

            print("Model", h5_files[int(ans) - 1], "is loaded")
            return model
        else:
            print("Invalid choice")
            exit()

    def get_training_mode(self):
        print("\nShould the model be trained? \n1) Train \n2) Don't train")
        ans = input('Choice: ')
        if ans == "1":
            print("The model will be trained")
            return True
        elif ans == "2":
            print("The model will not be trained")
            return False
        else:
            print("Invalid choice")
            exit()

    def create_model(self):
        model = Sequential()

        model.add(Dense(256, activation='relu', input_dim=self.n_inputs))
        model.add(Dropout(0.15))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(self.n_outputs))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['mae'])
        return model

    def update(self, npc):
        if self.training_mode:
            self.update_memory(npc)
            self.train()
            # todo: should the call below be removed?
            self.get_action(
                npc)  # Why is this being called here???????? does the agent give a new action without the npc asking for one?

    def update_memory(self, npc):
        self.memory.append(np.array([npc.old_state, npc.new_state, npc.action, npc.reward, npc.terminal_state]))
        return

        # The memory model which separates the state based on actions as to not batch train on different q values
        for i in range(len(self.memory)):
            if npc.old_state.tolist() not in self.memory[i][0]:
                self.memory[i].append(
                    np.array([npc.old_state, npc.new_state, npc.action, npc.reward, npc.terminal_state]))
                self.memory[i][0].append(npc.old_state.tolist())
                self.memory[i][1].append(npc.action)
                return

            elif npc.old_state.tolist() in self.memory[i][0] and npc.action == self.memory[i][1][
                self.memory[i][0].index(npc.old_state.tolist())]:
                self.memory[i].append(
                    np.array([npc.old_state, npc.new_state, npc.action, npc.reward, npc.terminal_state]))
                self.memory[i][0].append(npc.old_state.tolist())
                self.memory[i][1].append(npc.action)
                return

        # If not yet added to any memory list start a new memory list
        new_memory_list = [[npc.old_state.tolist()], [npc.action],
                           np.array([npc.old_state, npc.new_state, npc.action, npc.reward, npc.terminal_state])]
        self.memory.append(new_memory_list)

    def train(self):
        """
        index = np.random.randint(0, len(self.memory))
        train_batch = self.memory[index][2:len(self.memory[index]) + 1]

        old_states = np.array([x[0] for x in train_batch])
        old_qs_list = self.model.predict(old_states)

        new_states = np.array([x[1] for x in train_batch])
        new_qs_list = self.target_model.predict(new_states)
        """
        if len(self.memory) > self.training_batch_size:
            train_batch = random.sample(self.memory, self.training_batch_size)
        else:
            return

        old_states = np.array([transition[0] for transition in train_batch]).reshape(-1, self.n_inputs)
        old_qs_list = self.model.predict(old_states)

        new_states = np.array([transition[1] for transition in train_batch]).reshape(-1, self.n_inputs)
        new_qs_list = self.model.predict(new_states)

        for i, (old_state, new_state, action, reward, in_terminal_state) in enumerate(train_batch):
            if not in_terminal_state:
                max_new_q = np.max(new_qs_list[i])
                updated_q = reward + self.discount * max_new_q
            else:
                updated_q = reward

            old_qs = old_qs_list[i]
            old_qs[action] = updated_q

            self.model.fit(np.array(old_state).reshape(-1, self.n_inputs), np.array(old_qs).reshape(-1, self.n_outputs), epochs=1,
                           verbose=self.verbose, shuffle=False)


        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        self.save_counter += 1
        if self.save_counter >= self.save_max:
            self.save_counter = 0
            self.save_model()

    def get_action(self, npc):
        # Sometime return a random action
        self.epsilon *= self.epsilon_decay

        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_max
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_outputs)

        state = np.array([npc.new_state])
        return np.argmax(self.model.predict(state))

    def save_model(self, name=False):
        if not name:
            name = str(int(time.time()))

        self.model.save("./trained_agents/" + name + ".h5")

    def eval(self):
        states = [[0], [1]]

        while len(states[0]) < 8:
            tmp = []
            for state in states:

                for i in range(2):
                    x = state.copy()
                    x.append(i)
                    tmp.append(x)

            states = tmp

        for state in states:
            if state[4] == 1:
                state[6] = 0
            if state[5] == 1:
                state[7] = 0

        # why are the duplicates not removed?
        tmp = []
        for state in states:
            if state not in tmp:
                tmp.append(state)
        states = tmp

        states = np.array(states)
        predictions = self.model.predict(states)
        n_wrong_predictions = 0

        for i in range(len(states)):
            if states[i][np.argmax(predictions[i])] == 1:
                n_wrong_predictions += 1
        print("wrongs predictions:", n_wrong_predictions / len(states), " %")
