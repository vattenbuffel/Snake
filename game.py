import pygame
import time
import npc
import food

#todo: add a method too check the state of the game
class Game:
    def __init__(self):
        self.display_width = 1280
        self.display_height = 720
        self.pygame = pygame
        self.gameDisplay = pygame.display.set_mode((self.display_width, self.display_height))
        self.pygame.display.set_caption('A society')
        self.clock = pygame.time.Clock()
        self.white = (255, 255, 255)
        self.fps = 60
        self.is_done = False
        self.npcs = []
        self.food = []
        self.inhabit_society(n_npc=50, n_food=10)

    def update(self):
        self.handle_event()
        for kirby in self.npcs:
            kirby.update(self)

        for cookie in self.food:
            cookie.update()

        self.handle_dead()

    def handle_event(self):
        for event in self.pygame.event.get():
            if event.type == pygame.KEYUP or event.type == pygame.KEYDOWN:
                self.read_keyboard(event)

    def read_keyboard(self, event):
        if event.key == self.pygame.K_ESCAPE:
            self.is_done = True

    def render(self):
        self.gameDisplay.fill(self.white)
        for kirby in self.npcs:
            try:
                kirby.render()
            except:
                self.npcs.remove(kirby)
                break
        for cookie in self.food:
            try:
                cookie.render()
            except:
                self.npcs.remove(cookie)
                break

        self.pygame.display.update()

    def run(self):
        while not self.is_done:
            self.update()
            self.render()
            self.clock.tick(self.fps)

        self.pygame.quit()

    def handle_dead(self):
        alive = [kirby if kirby.alive else -1 for kirby in self.npcs]
        try:
            alive.remove(-1)
        except:
            pass
        self.npcs = alive

        self.food = [cookie if cookie.alive else food.Food(pygame.image.load('./img/food.jpg'), self) for cookie in self.food]


    def inhabit_society(self, n_npc, n_food):
        self.npcs = [npc.NPC(self, male=i%2==0) for i in range(n_npc)]
        self.food = [food.Food(pygame.image.load('./img/food.jpg'), self) for i in range(n_food)]

Game().run()

quit()
