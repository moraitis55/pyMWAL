import csv
import os

from PIL import Image
import numpy as np
import cv2

# from gridworld.blob import Blob
from blob import Blob

class GridState:
    def __init__(self, player_position, food_position, enemy_position, action=None):
        self.pp = player_position  # type: tuple
        self.fp = food_position  # type: tuple
        self.ep = enemy_position  # type: tuple
        if action is not None:
            self.ac = action

    def __str__(self):
        if hasattr(self, 'ac'):
            return "player{0}, food{1}, enemy{2}, action{3}".format(self.pp, self.fp, self.ep, self.ac)
        else:
            return "player{0}, food{1}, enemy{2}".format(self.pp, self.fp, self.ep)


class GridEnv:
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def __init__(self, episode_steps=200, size=10, allow_vertical_movement=False, move_penalty=-1, enemy_penalty=-300,
                 food_reward=25, return_images=True, enable_enemy_move=False, enable_food_move=False, dizzy=False):
        self.dizzy = dizzy
        self.size = size
        self.episode_steps = episode_steps
        self.vertical_movement = allow_vertical_movement
        self.move_penalty = move_penalty
        self.enemy_penalty = enemy_penalty
        self.food_reward = food_reward
        self.enemy_move = enable_enemy_move
        self.food_move = enable_food_move
        self.return_images = return_images
        self.observation_space_values = (size, size, 3)  # 4
        self.feature_nr = 6  # todo: change in to parameter.
        if allow_vertical_movement:
            self.action_space_size = 9
        else:
            self.action_space_size = 4
        self.state_space_size, self.state_space_index = self.create_state_space()
        self.action_state_index = self.create_action_state_index()

    def create_state_space(self, include_absorbing_state=False):
        """
        Creates the state space index of the grid world environment.
        :param add_absorbing_state: Used to create indexes including one more state (the absorbing state s0) used by mwal algorithm
        :return: total number of states, state space index
        """

        state_space_index = {}
        i = 0  # type: int
        if include_absorbing_state:
            print("Initializing environment space including the absorbing state..")
        else:
            print("Initializing environment space..")
        for x1 in range(self.size):
            for y1 in range(self.size):
                for x2 in range(self.size):
                    for y2 in range(self.size):
                        for x3 in range(self.size):
                            for y3 in range(self.size):
                                st = GridState(
                                    player_position=(x1, y1),
                                    food_position=(x2, y2),
                                    enemy_position=(x3, y3)
                                )
                                if not st.fp == st.ep:
                                    state_space_index[st.__str__()] = i
                                    i += 1
        if include_absorbing_state:
            st = GridState(
                player_position=(0, 0),
                food_position=(0, 0),
                enemy_position=(0, 0)
            )
            state_space_index[st.__str__()] = i
            i += 1

        return state_space_index.__len__(), state_space_index

    def create_action_state_index(self, include_absorbing_state=False):
        """
        Creates the action-state couples index of the grid world environment.
        :param add_absorbing_state: Used to create indexes including one more state (the absorbing state s0) used by mwal algorithm
        :return: state-action space index
        """

        action_state_pair_index = {}
        i = 0  # type: int
        if include_absorbing_state:
            print("Initializing environment action-state indexes (including the absorbing state)..")
        else:
            print("Initializing environment action-state indexes..")
        for a in range(self.action_space_size):
            for x1 in range(self.size):
                for y1 in range(self.size):
                    for x2 in range(self.size):
                        for y2 in range(self.size):
                            for x3 in range(self.size):
                                for y3 in range(self.size):
                                    st = GridState(
                                        player_position=(x1, y1),
                                        food_position=(x2, y2),
                                        enemy_position=(x3, y3),
                                        action=a
                                    )
                                    if not st.fp == st.ep:
                                        action_state_pair_index[st.__str__()] = i
                                        i += 1
            if include_absorbing_state:
                st = GridState(
                    player_position=(0, 0),
                    food_position=(0, 0),
                    enemy_position=(0, 0),
                    action=a
                )
                action_state_pair_index[st.__str__()] = i
                i += 1

        return action_state_pair_index

    def create_absorbing_state_indexes(self):
        self.state_space_size, self.state_space_index = self.create_state_space(include_absorbing_state=True)
        self.action_state_index = self.create_action_state_index(include_absorbing_state=True)

    def write_out_indexing(self):
        file = os.path.join('gridworld', 'state_index_mapping.csv')
        if not os.path.exists(file):
            print("Writing out state space indexing..")
            with open(file, 'w') as f:
                writer = csv.writer(f)
                import operator
                dict = self.state_space_index
                x = sorted(dict.items(), key=operator.itemgetter(1))
                import collections
                sorted_dict = collections.OrderedDict(x)
                for key in sorted_dict:
                    writer.writerow([sorted_dict[key], key])
            print("Done")

    def reset(self):
        self.player = Blob(self.size, self.vertical_movement, self.dizzy)
        self.food = Blob(self.size, self.vertical_movement, False)
        while self.food == self.player:
            self.food = Blob(self.size, self.vertical_movement, False)
        self.enemy = Blob(self.size, self.vertical_movement, False)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.size, self.vertical_movement, False)

        self.episode_step = 0

        if self.return_images:
            observation = np.array(self.get_image())
        else:
            observation = (self.player - self.food, self.player - self.enemy)
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
            new_observation = (self.player - self.food, self.player - self.enemy)

        if self.player == self.enemy:
            reward = self.enemy_penalty
        elif self.player == self.food:
            reward = self.food_reward
        else:
            reward = self.move_penalty

        done = False
        if reward == self.food_reward or reward == self.enemy_penalty or self.episode_step == self.episode_steps - 1:
            done = True

        return new_observation, reward, done

    def render(self, wait=1):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(wait)

    # FOR CNN #
    def get_image(self):
        env = np.zeros(self.observation_space_values, dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img
