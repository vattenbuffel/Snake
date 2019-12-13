import pygame
import random
import math


class NPC:
    def __init__(self, game, size=1, male=True):
        self.max_x = game.display_width
        self.max_y = game.display_height
        self.disp = game.gameDisplay
        self.size = [size]
        self.img_heigt = int(50 * self.size[0])
        self.img_width = int(50 * self.size[0])

        self.state_values = {'food_located': False, 'pregnant': False, 'hungry': False, 'full': True,
                             'mate_located': False, 'mating': False,
                             'eating': False, 'null': False}
        self.states = {'food_located': False, 'pregnant': False, 'hungry': False, 'full': True,
                             'mate_located': False, 'mating': False,
                             'eating': False, 'null': False}
        hungry_kirby = pygame.transform.smoothscale(pygame.image.load('./img/blue_kirby.jpg'),
                                                    (self.img_width, self.img_width))
        full_kirby = pygame.transform.smoothscale(pygame.image.load('./img/pink_kirby.png'),
                                                  (self.img_width, self.img_width))
        food_located_kirby = pygame.transform.smoothscale(pygame.image.load('./img/kirby_suck.jpg'),
                                                          (self.img_width, self.img_width))
        mate_located_kirby = pygame.transform.smoothscale(pygame.image.load('./img/orange_kirby.png'),
                                                          (self.img_width, self.img_width))
        mating_kirby = pygame.transform.smoothscale(pygame.image.load('./img/orange_kirby.png'),
                                                    (self.img_width, self.img_width))
        pregnant_kirby = pygame.transform.smoothscale(pygame.image.load('./img/pink_kirby.png'),
                                                    (self.img_width, self.img_width))
        self.images = {0: food_located_kirby, 1:pregnant_kirby, 2: hungry_kirby,
                       3: full_kirby, 4: mate_located_kirby,
                       5: mating_kirby, 6: food_located_kirby}

        self.hunger = 0
        self.target_food = None
        self.target_mate = None
        self.alive = True
        self.x = random.randint(0, self.max_x - self.img_width)
        self.y = random.randint(0, self.max_y - self.img_heigt)
        self.dir = random.randint(0, 360 / 10) * 10
        self.dir_factor = 36
        self.out_of_bound = False #This should be changed to a state. #todo change to state
        self.male = male
        self.pregnancy_counter = 100

        self.bite_size = 0.25*self.size[0]**0.7
        self.step_size = 7*(self.size[0]*5+1)**-0.5
        self.sex_appeal = 0.01 *self.size[0]
        self.search_distance = 300 * self.size[0]

    def eat(self):
        if self.state_values['eating']:
            self.hunger -= self.target_food.get_eaten(self.bite_size)

    def render(self):

        # todo: find a good way to get the right picture
        """
        img = self.images['full']
        if self.state_values['mating']:
        """
        img = self.images[3]
        self.disp.blit(img, (self.x, self.y))

    def update(self, game):
        self.hunger += 0.1*self.size[0]**0.7
        if self.hunger > 100:
            self.alive = False
            print('A death occured. A kirby with size',self.size[0], "died. It's was a male", self.male)

        self.update_state(game)
        self.action(game)

    def action(self, game):
        self.move()
        self.eat()
        self.reproduce(game=game)

    # Searches the game for food and mates
    def search(self, game):
        # Make sure it goes for the nearest food
        # What's the order, food>sex?
        # make sure that it searches from it's own base, is middle

        # find mate
        if self.state_values['full']:
            closest_mate = None
            closest_dist = (self.max_x ** 2 + self.max_y ** 2)

            npc_x, npc_y = self.x + self.img_width / 2, self.y + self.img_heigt / 2
            for kirby in game.npcs:
                # Sometime it bugs out and kirby is just a int, this will crash the program
                if type(kirby)==int:
                    game.npcs.remove(kirby)
                    return
                # If it's already pregnant or same sex, don't bother, or not full
                if not kirby.male == self.male and not kirby.state_values['pregnant'] and kirby.state_values['full']:
                    kirby_x, kirby_y = kirby.x + kirby.img_width / 2, kirby.y + kirby.img_heigt / 2
                    dist = ((kirby_x - npc_x) ** 2 + (kirby_y - npc_y) ** 2) ** 0.5
                    if dist < closest_dist and dist < self.search_distance:
                        closest_dist = dist
                        closest_mate = kirby

                        # If touching the mate have a chance to impregnate it, else just move towards it
                        if dist < ((self.img_heigt / 2) ** 2 + (self.img_width / 2) ** 2) ** 0.5:
                            self.state_values['mating'] = True
                        else:
                            self.state_values['mating'] = False
                            self.state_values['mate_located'] = True

            if closest_mate is not None:
                y_dif = ((closest_mate.y + closest_mate.img_heigt / 2) - npc_y)
                x_dif = ((closest_mate.x + closest_mate.img_width / 2) - npc_x)
                try:
                    self.dir = math.degrees(math.atan2(y_dif, x_dif))
                except:
                    sign = lambda a: (a > 0) - (a < 0)
                    self.dir = sign(y_dif) * 180
                self.target_mate = closest_mate
                return

        # It's not found  mate or it's hungry so these states must be false
        self.state_values['mating'] = False
        self.state_values['mate_located'] = False
        # find food
        closest_food = None
        closest_dist = (self.max_x ** 2 + self.max_y ** 2)

        npc_x, npc_y = self.x + self.img_width / 2, self.y + self.img_heigt / 2
        for food in game.food:
            food_x, food_y = food.x + food.img_width / 2, food.y + food.img_heigt / 2
            dist = ((food_x - npc_x) ** 2 + (food_y - npc_y) ** 2) ** 0.5
            if dist < closest_dist and dist < self.search_distance:
                closest_dist = dist
                closest_food = food

                # If touching the food eat it, else just move towards it
                if dist < ((self.img_heigt / 2) ** 2 + (self.img_width / 2) ** 2) ** 0.5:
                    self.state_values['eating'] = True
                else:
                    self.state_values['eating'] = False
                    self.state_values['food_located'] = True

        if closest_food is not None:
            self.target_food = closest_food
            y_dif = ((closest_food.y + closest_food.img_heigt / 2) - npc_y)
            x_dif = ((closest_food.x + closest_food.img_width / 2) - npc_x)
            try:
                self.dir = math.degrees(math.atan2(y_dif, x_dif))
            except:
                sign = lambda a: (a > 0) - (a < 0)
                self.dir = sign(y_dif) * 180

            return

        self.state_values['eating'] = False
        self.state_values['food_located'] = False

    def update_state(self, game):
        if self.hunger > 25:
            self.state_values['hungry'] = True
            self.state_values['full'] = False
        else:
            self.state_values['full'] = True
            self.state_values['hungry'] = False

        self.search(game)

    def move(self):
        # If it's moved out of bound move it back in and give it a new dir
        # there's always a chance to move in a new dir
        prob_to_change_dir = 0.01
        first_time = True

        # If food's found there's no chance to move in a random dir
        if self.state_values['food_located'] or self.state_values['mate_located']:
            prob_to_change_dir = 0

        while self.out_of_bound or first_time:
            first_time = False
            if random.random() < prob_to_change_dir or self.out_of_bound:
                self.dir = random.randint(0, (360 - self.dir_factor) / self.dir_factor) * self.dir_factor

            x_step = math.cos(math.radians(self.dir)) * self.step_size
            self.x = self.x + x_step
            self.y = self.y + math.sin(math.radians(self.dir)) * self.step_size
            self.out_of_bound_handler()

    # Handles out of bound
    def out_of_bound_handler(self):
        self.out_of_bound = False
        if self.x + self.img_width > self.max_x:
            self.x = self.max_x - self.img_width
            self.out_of_bound = True

        elif self.x < 0:
            self.x = 0
            self.out_of_bound = True

        if self.y + self.img_heigt > self.max_y:
            self.y = self.max_y - self.img_heigt
            self.out_of_bound = True

        elif self.y < 0:
            self.y = 0
            self.out_of_bound = True

    def reproduce(self, game=None, sex_appeal=0):
        if self.state_values['mating']:
            # If male try to impregnate, if female just wait
            if self.male and not self.target_mate.state_values['pregnant']:
                self.target_mate.reproduce(sex_appeal = self.sex_appeal)
            else:
                if random.random() < self.sex_appeal + sex_appeal:
                    self.state_values['pregnant'] = True

        elif self.state_values['pregnant']:
            self.pregnancy_counter -= 1
        if self.pregnancy_counter < 0:
            self.state_values['pregnant'] = False
            self.pregnancy_counter = 100
            baby_size = random.randint(-30,30)/100 + self.size[0]
            if baby_size <= 0:
                baby_size = self.size[0]
            male = random.random()<0.5
            game.npcs.append(NPC(game, size = baby_size, male = male))
            print('A BIRTH with size',baby_size, "happend. It's a male", male)



