import random

import numpy as np


class Blob:
    def __init__(self, size, vertical_movement, dizzy, posx=None, posy=None):
        self.size = size
        self.dizzy = dizzy
        self.x = posx if posx else np.random.randint(0, size)
        self.y = posy if posy else np.random.randint(0, size)
        self.vertical_movement = vertical_movement

    def __str__(self):
        return "Blob ({0}, {1})".format(self.x, self.y)

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        if self.dizzy:
            num = random.randint(1, 10)
            if num > 6:
                moves_nr = 9 if self.vertical_movement else 4
                new_choice_list = [x for x in range(moves_nr) if x != choice]
                choice = random.choice(new_choice_list)
                self.set_action(choice)
            else:
                self.set_action(choice)

        else:
            self.set_action(choice)

    def set_action(self, choice):
        """
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        """
        # if vertical movement is true enable it.
        if self.vertical_movement:
            if choice == 0:
                self.move(x=1, y=1)
            elif choice == 1:
                self.move(x=-1, y=-1)
            elif choice == 2:
                self.move(x=-1, y=1)
            elif choice == 3:
                self.move(x=1, y=-1)
            elif choice == 4:
                self.move(x=1, y=0)
            elif choice == 5:
                self.move(x=-1, y=0)
            elif choice == 6:
                self.move(x=0, y=1)
            elif choice == 7:
                self.move(x=0, y=-1)
            elif choice == 8:
                self.move(x=0, y=0)
        else:
            if choice == 0:
                self.move(x=1, y=1)
            elif choice == 1:
                self.move(x=-1, y=-1)
            elif choice == 2:
                self.move(x=-1, y=1)
            elif choice == 3:
                self.move(x=1, y=-1)


    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size - 1:
            self.x = self.size - 1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size - 1:
            self.y = self.size - 1
