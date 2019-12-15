import pygame
import random
import math
import tensorflow
from keras.models import Sequential
import numpy as np


class NPC:
    def __init__(self, game):
        self.max_x = game.n_squares_width - 1
        self.max_y = game.n_squares_height - 1
        self.alive = True
        self.squares = game.squares
        self.xs = [int(self.max_x / 2)]
        self.ys = [int(self.max_y / 2)]
        self.human_playing = game.human_playing
        self.dir = "NORTH"
        self.frame_delay_max = 30
        self.frame_delay = self.frame_delay_max
        self.eaten = False
        self.end_pos = None
        self.old_distance_to_food = -1
        self.new_distance_to_food = -1
        self.old_state = [-1 for x in game.squares]
        self.new_state = [-1 for x in game.squares]
        self.reward = -1

    def eat(self, game):
        if self.xs[0] == game.food.x and self.ys[0] == game.food.y:
            self.eaten = True
            self.end_pos = (self.xs[-1], self.ys[-1])
            game.food.get_eaten()

    def reproduce(self):
        if self.eaten and self.frame_delay == 0:
            x, y = self.end_pos
            self.xs.append(x)
            self.ys.append(y)
            self.eaten = False

    def render(self):
        pass

    def update_state(self, game):
        self.old_state = self.new_state
        self.old_state = np.array([x.state for x in game.squares])

    def update_dist(self, game):
        self.old_distance_to_food = self.new_distance_to_food
        self.new_distance_to_food = self.calc_dist_to_food(game)

    def update_reward(self, game):
        # If snake died reward is -10^6
        if not self.alive:
            self.reward = -10**6

        # If food has been eaten reward is 10^3
        elif self.eaten:
            self.reward = 10 ** 3

        # Check if gotten closer to food reward is 1 and update old_dist_to_food
        elif self.new_distance_to_food < self.old_distance_to_food:
            self.reward = 1
            # If not closer reward is -1
        else:
            self.reward = -1

    def calc_dist_to_food(self, game):
        x_diff = self.xs[0] - game.food.x
        y_diff = self.ys[0] - game.food.y
        dist = (x_diff**2 + y_diff**2)**0.5
        return dist

    def update(self, game):
        self.frame_delay -= 1
        self.eat(game)
        self.move(game.events)

        self.die()
        self.reproduce()
        self.update_dist(game)
        self.update_state(game)
        self.update_reward(game)
        self.square_update()
        if self.frame_delay <= 0:
            self.frame_delay = self.frame_delay_max

    def square_update(self):
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
                        self.dir = "NORTH"
                    if event.key == pygame.K_s:
                        self.dir = "SOUTH"
                    if event.key == pygame.K_a:
                        self.dir = "WEST"
                    if event.key == pygame.K_d:
                        self.dir = "EAST"

        if self.frame_delay == 0:
            for i in range(len(self.xs) - 1):
                self.xs[len(self.xs) - i - 1] = self.xs[len(self.xs) - i - 2]
                self.ys[len(self.ys) - i - 1] = self.ys[len(self.ys) - i - 2]

            if self.dir == "NORTH":
                self.ys[0] = self.ys[0] - 1
            elif self.dir == "EAST":
                self.xs[0] = self.xs[0] + 1
            elif self.dir == "SOUTH":
                self.ys[0] = self.ys[0] + 1
            elif self.dir == "WEST":
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
                    print('died because of overlap')
                    return
