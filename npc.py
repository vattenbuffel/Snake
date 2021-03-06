import pygame
import random
import math
import numpy as np
import Q_learning
from collections import deque
import time


class NPC:
    def __init__(self, game):
        self.max_x = game.n_squares_width - 1
        self.max_y = game.n_squares_height - 1
        self.alive = True
        self.squares = game.squares
        self.xs = [int(self.max_x / 2)]
        self.ys = [int(self.max_y / 2)]
        self.human_playing = game.human_playing
        self.dirs = {"NORTH": 0, "EAST": 1, "SOUTH": 2, "WEST": 3}
        self.dir = self.dirs["NORTH"]
        self.frame_delay_max = 30 if self.human_playing else 1
        self.frame_delay = self.frame_delay_max
        self.eaten = False
        self.end_pos = None
        self.score = 0
        self.score_mem = deque(maxlen=100)
        self.avg_score = 0

        self.human_playing = game.human_playing
        # Initiate the q-learning stuff
        self.old_distance_to_food = -1
        self.new_distance_to_food = -1
        self.old_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.new_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.reward = -1
        self.square_update()
        self.update_state(game)
        self.update_dist(game)
        self.update_reward(game)
        self.terminal_state = False

        self.n_actions = 4
        self.action = self.dir
        if not self.human_playing:
            self.agent = Q_learning.Agent(self)

    def eat(self, game):
        if self.xs[0] == game.food.x and self.ys[0] == game.food.y:
            self.eaten = True
            self.end_pos = (self.xs[-1], self.ys[-1])
            game.food.get_eaten()

    def reproduce(self):
        if self.eaten and self.frame_delay == 0:
            self.score += 1
            x, y = self.end_pos
            self.xs.append(x)
            self.ys.append(y)

    def render(self):
        pass

    def update_state(self, game):
        self.old_state = self.new_state

        def remove_false(x):
            if False in x:
                x = list(set(x))
                x.remove(False)
            return x

        indices = [i if self.ys[0] - 1 == self.ys[i] else False for i in range(len(self.xs))]
        indices = remove_false(indices)
        danger_NORTH = self.xs[0] in np.array(self.xs)[indices] or self.ys[0] == 0

        indices = [i if self.xs[0] + 1 == self.xs[i] else False for i in range(len(self.xs))]
        indices = remove_false(indices)
        danger_EAST = self.ys[0] in np.array(self.ys)[indices] or self.xs[0] == self.max_x

        indices = [i if self.ys[0] + 1 == self.ys[i] else False for i in range(len(self.xs))]
        indices = remove_false(indices)
        danger_SOUTH = self.xs[0] in np.array(self.xs)[indices] or self.ys[0] == self.max_y

        indices = [i if self.xs[0] - 1 == self.xs[i] else False for i in range(len(self.xs))]
        indices = remove_false(indices)
        danger_WEST = self.ys[0] in np.array(self.ys)[indices] or self.xs[0] == 0

        food_NORTH = self.ys[0] > game.food.y
        food_EAST = self.xs[0] < game.food.x
        food_SOUTH = self.ys[0] < game.food.y
        food_WEST = self.xs[0] > game.food.x

        going_NORTH = 0 == self.dir
        going_EAST = 1 == self.dir
        going_SOUTH = 2 == self.dir
        going_WEST = 3 == self.dir

        self.new_state = [danger_NORTH, danger_EAST, danger_SOUTH, danger_WEST, food_NORTH, food_EAST, food_SOUTH,
                          food_WEST, going_NORTH, going_EAST, going_SOUTH, going_WEST]

        self.new_state = np.array([1 if condition else 0 for condition in self.new_state])

    def update_dist(self, game):
        self.old_distance_to_food = self.new_distance_to_food
        self.new_distance_to_food = self.calc_dist_to_food(game)

    def update_reward(self, game):
        # If snake died reward is -10^6
        if not self.alive:
            self.reward = -100

        # Should never happen
        elif not (False in [game.squares[0].states['board'] == x.state for x in game.squares]):
            self.reward = 10 ** 7
            print('Beat the game baby')
            self.alive = False

        # If food has been eaten reward is 10^3
        elif self.new_distance_to_food == 0:
            self.reward = 25

        # Check if gotten closer to food reward is 1 and update old_dist_to_food
        elif self.new_distance_to_food < self.old_distance_to_food:
            self.reward = 1
            # If not closer reward is -1
        else:
            self.reward = -1

    def calc_dist_to_food(self, game):
        x_diff = self.xs[0] - game.food.x
        y_diff = self.ys[0] - game.food.y
        dist = (x_diff ** 2 + y_diff ** 2) ** 0.5
        return dist

    def update(self, game):
        self.frame_delay -= 1
        self.eat(game)
        self.move(game.events)

        self.die()
        self.reproduce()
        self.square_update()
        self.update_qlearning_stuff(game)

        if self.frame_delay <= 0:
            self.frame_delay = self.frame_delay_max
            self.eaten = False
            #tmp
            #self.update_state(game)

    def update_qlearning_stuff(self, game):
        if not self.human_playing and self.frame_delay <= 0:
            self.update_dist(game)
            self.update_state(game)
            self.update_reward(game)
            self.terminal_state = not self.alive
            self.action = self.dir
            self.agent.update(self)

    def square_update(self):
        if not self.alive:
            return

        for i in range(len(self.xs)):
            index = self.xs[i] * (1 + self.max_x) + self.ys[i]
            self.squares[index].state = self.squares[0].states['snake']

        index = self.xs[0] * (1 + self.max_x) + self.ys[0]
        self.squares[index].state = self.squares[0].states['snake_head']

    def move(self, events):
        if self.human_playing:
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        self.dir = self.dirs["NORTH"]
                    if event.key == pygame.K_s:
                        self.dir = self.dirs["SOUTH"]
                    if event.key == pygame.K_a:
                        self.dir = self.dirs["WEST"]
                    if event.key == pygame.K_d:
                        self.dir = self.dirs["EAST"]

        elif not self.human_playing and self.frame_delay == 0:
            self.dir = self.agent.get_action(self)

        if self.frame_delay == 0:
            for i in range(len(self.xs) - 1):
                self.xs[len(self.xs) - i - 1] = self.xs[len(self.xs) - i - 2]
                self.ys[len(self.ys) - i - 1] = self.ys[len(self.ys) - i - 2]

            if self.dir == self.dirs["NORTH"]:
                self.ys[0] = self.ys[0] - 1
            elif self.dir == self.dirs["EAST"]:
                self.xs[0] = self.xs[0] + 1
            elif self.dir == self.dirs["SOUTH"]:
                self.ys[0] = self.ys[0] + 1
            elif self.dir == self.dirs["WEST"]:
                self.xs[0] = self.xs[0] - 1

    def die(self):
        if self.xs[0] < 0 or self.xs[0] > self.max_x:
            self.alive = False
        if self.ys[0] < 0 or self.ys[0] > self.max_y:
            self.alive = False

        for i in range(len(self.xs)):
            for j in range(len(self.xs)):
                if i == j:
                    continue
                if self.xs[i] == self.xs[j] and self.ys[i] == self.ys[j]:
                    self.alive = False
                    return

    def restart(self, game):
        self.xs = [int(self.max_x / 2)]
        self.ys = [int(self.max_y / 2)]
        self.alive = True
        self.square_update()
        self.old_distance_to_food = -1
        self.new_distance_to_food = self.calc_dist_to_food(game)
        self.old_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.new_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.reward = -1
        self.square_update()
        self.update_state(game)
        self.update_dist(game)
        self.update_reward(game)
        self.score_mem.append(self.score)
        self.avg_score = sum(self.score_mem)/len(self.score_mem)
        print('Died with score:', self.score)
        print("average score:", self.avg_score)
        print()

        # save if the model is better than the avg
        if self.score > self.avg_score:
            a = time.localtime()
            name = str(str(a.tm_hour) + "_" + str(a.tm_min) + "_" + str(a.tm_sec))+"score_"+str(self.score)
            self.agent.save_model(name)

        self.score = 0
        self.terminal_state = False
