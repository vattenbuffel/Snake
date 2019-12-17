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

        #self.spawn(game)

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
        #self.x = self.y = 0
        #return
        # See if snake is at this pos
        x_in_snake = self.x in game.snake.xs
        if x_in_snake and self.y == game.snake.ys[game.snake.xs.index(self.x)]:
            try:
                self.spawn(game)
            except:
                print('wallah')

        self.alive = True

    def get_eaten(self):
        self.die()
