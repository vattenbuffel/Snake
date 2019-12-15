import pygame
import random


class Food:
    def __init__(self, game):
        self.max_x = game.n_squares_width - 1
        self.max_y = game.n_squares_height - 1
        self.alive = True
        self.squares = game.squares
        self.x = -1
        self.y = -1
        self.size = 20

        self.spawn(game)

    def render(self):
        pass

    def update(self, game):
        if not self.alive:
            self.spawn(game)
        index = self.x * (1 + self.max_x) + self.y
        self.squares[index].state = self.squares[0].states['food']

    def die(self):
        self.alive = False

    def spawn(self, game):
        self.x = random.randint(0, self.max_x)
        self.y = random.randint(0, self.max_y)
        if self.x in game.snake.xs and self.y in game.snake.ys:
            try:
                self.spawn(game)
            except:
                print('wallah')

        self.alive = True

    def get_eaten(self):
        self.die()
