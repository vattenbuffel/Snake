import pygame


# todo: make pictures see through
class Square:
    def __init__(self, x, y, width, height, game):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.disp = game.gameDisplay
        self.scale_factor = game.img_scale_factor
        self.states = {'board': 0, 'snake': 1, 'food': 2, 'snake_head': 3}
        self.state = self.states['board']

        board_img = pygame.image.load('./img/board.jpg')
        board_img = pygame.transform.smoothscale(board_img, (
            int(self.width * self.scale_factor), int(self.height * self.scale_factor)))

        food_img = pygame.image.load('./img/food.jpg')
        food_img = pygame.transform.smoothscale(food_img, (
            int(self.width * self.scale_factor), int(self.height * self.scale_factor)))

        snake_img = pygame.image.load('./img/snake.jpg')
        snake_img = pygame.transform.smoothscale(snake_img, (
            int(self.width * self.scale_factor), int(self.height * self.scale_factor)))

        snake_head_img = pygame.image.load('./img/snake_head.jpg')
        snake_head_img = pygame.transform.smoothscale(snake_head_img, (
            int(self.width * self.scale_factor), int(self.height * self.scale_factor)))

        self.images = {self.states['board']: board_img, self.states['snake']: snake_img,
                       self.states['snake_head']: snake_head_img, self.states['food']: food_img}
        self.img = None

    def update(self):
        self.state = self.states['board']

    def render(self):
        self.img = self.images[self.state]
        x = self.x * self.width + self.width * (1 - self.scale_factor) / 2
        y = self.y * self.height + self.height * (1 - self.scale_factor) / 2
        self.disp.blit(self.img, (x, y))
