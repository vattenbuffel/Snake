import pygame
import random
import math


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

    def update(self, game):
        self.frame_delay -= 1
        self.eat(game)
        self.move(game.events)

        if self.eaten and self.frame_delay == 0:
            self.frame_delay_max -= 1
            if self.frame_delay_max <= 0:
                self.frame_delay_max = 1

        self.reproduce()
        self.die()
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
