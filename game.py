import pygame
import time
import npc
import food
import square

class Game:
    def __init__(self):
        self.display_width = 400
        self.display_height = 400
        self.n_squares_width = 7
        self.n_squares_height = 7
        self.pygame = pygame
        self.gameDisplay = pygame.display.set_mode((self.display_width, self.display_height))
        self.pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.black = (0, 0, 0)
        self.fps = 100
        self.is_done = False
        self.img_scale_factor = 0.9
        self.squares = [square.Square(x, y, self.display_width/self.n_squares_width, self.display_height/self.n_squares_height, self) for x in range(self.n_squares_width) for y in range(self.n_squares_height)]

        self.events = None
        self.human_playing = False

        #create food and spawn it
        self.food = food.Food(self)
        self.food.x = 0
        self.food.y = 0
        self.squares[0].state = self.squares[0].states['food']

        self.snake = npc.NPC(self)

    def update(self):
        self.handle_event()
        for sq in self.squares:
            sq.update()
        self.food.update(self)
        self.snake.update(self)

        if not self.snake.alive:
            self.restart()

    def handle_event(self):
        self.events = self.pygame.event.get()
        for event in self.events:
            if event.type == pygame.KEYUP or event.type == pygame.KEYDOWN:
                self.read_keyboard(event)

    def read_keyboard(self, event):
        if event.key == self.pygame.K_ESCAPE:
            self.is_done = True
        elif event.key == self.pygame.K_KP_MINUS:
            self.fps *= 0.75
            print('Changed fps to', self.fps)
        elif event.key == self.pygame.K_KP_PLUS:
            self.fps *= 0.75
            print('Changed fps to', self.fps)

    def render(self):
        self.gameDisplay.fill(self.black)
        for sq in self.squares:
            sq.render()

        self.pygame.display.update()

    def run(self):
        while not self.is_done:
            self.update()
            self.render()
            self.clock.tick(self.fps)

        self.pygame.quit()

    def restart(self):
        self.snake.restart(self)


Game().run()

