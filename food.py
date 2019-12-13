import pygame
import random


class Food:
    def __init__(self, img, game):
        self.max_x = game.display_width
        self.max_y = game.display_height
        self.disp = game.gameDisplay
        self.img_heigt = 40
        self.img_width = 40
        self.img = pygame.transform.smoothscale(img, (self.img_width, self.img_width))
        self.img = pygame.transform.rotate(self.img, -90)
        self.alive = True
        self.x = -1
        self.y = -1
        self.size = 20

        self.spawn()

    def render(self):
        self.disp.blit(self.img, (self.x, self.y))

    def update(self):
        pass

    def die(self):
        self.alive = False

    def spawn(self):
        self.x = random.randint(0, self.max_x - self.img_width)
        self.y = random.randint(0, self.max_y - self.img_heigt)
        self.alive = True

    def get_eaten(self, bite_size):
        # If the bite_size is bigger than what's left, return what's left and die. Else just return bite_size
        bite = bite_size
        if bite_size > self.size:
            bite = self.size
            self.die()

        self.size -= bite_size
        return bite
