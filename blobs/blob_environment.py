from blobs.blob import Blob
from PIL import Image
import numpy as np
import cv2


class BlobEnv:
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def __init__(self, size=10, allow_vertical_movement=False, move_penalty=1, enemy_penalty=-300,
                 food_reward=25, return_images=True, enable_enemy_move=False, enable_food_move=False):
        self.size = size
        self.vertical_movement = allow_vertical_movement
        self.move_penalty = move_penalty
        self.enemy_penalty = enemy_penalty
        self.food_reward = food_reward
        self.enemy_move = enable_enemy_move
        self.food_move = enable_food_move
        self.return_images = return_images
        self.observation_space_values = (size, size, 3)  # 4
        if allow_vertical_movement:
            self.action_space_size = 9
        else:
            self.action_space_size = 4

    def reset(self):
        self.player = Blob(self.size, self.vertical_movement)
        self.food = Blob(self.size, self.vertical_movement)
        while self.food == self.player:
            self.food = Blob(self.size, self.vertical_movement)
        self.enemy = Blob(self.size, self.vertical_movement)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.size, self.vertical_movement)

        self.episode_step = 0

        if self.return_images:
            observation = np.array(self.get_image())
        else:
            observation = (self.player - self.food) + (self.player - self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        if self.enemy_move:
         self.enemy.move()
        if self.food_move:
         self.food.move()

        if self.return_images:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player - self.food) + (self.player - self.enemy)

        if self.player == self.enemy:
            reward = -self.enemy_penalty
        elif self.player == self.food:
            reward = self.food_reward
        else:
            reward = -self.move_penalty

        done = False
        if reward == self.food_reward or reward == -self.enemy_penalty or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros(self.observation_space_values, dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img
