import pygame
import time
import npc
import food
import square

class Game:
    def __init__(self):
        self.display_width = 1280
        self.display_height = 720
        self.n_squares_width = 10
        self.n_squares_height = 10
        self.pygame = pygame
        self.gameDisplay = pygame.display.set_mode((self.display_width, self.display_height))
        self.pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.black = (0, 0, 0)
        self.fps = 60
        self.is_done = False
        self.img_scale_factor = 0.9
        self.squares = [square.Square(x, y, self.display_width/self.n_squares_width, self.display_height/self.n_squares_height, self) for x in range(self.n_squares_width) for y in range(self.n_squares_height)]

        self.events = None
        self.human_playing = True
        self.snake = npc.NPC(self)
        self.food = food.Food(self)

    def update(self):
        self.handle_event()
        for sq in self.squares:
            sq.update()
        self.food.update(self)
        self.snake.update(self)

        if not self.is_done:
            self.is_done = not self.snake.alive

    def handle_event(self):
        self.events = self.pygame.event.get()
        for event in self.events:
            if event.type == pygame.KEYUP or event.type == pygame.KEYDOWN:
                self.read_keyboard(event)

    def read_keyboard(self, event):
        if event.key == self.pygame.K_ESCAPE:
            self.is_done = True

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


Game().run()

quit()
