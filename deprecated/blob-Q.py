import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 20
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000

start_q_table = None  # or filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}

"""
Environment info:

observation = relative distance player-food, player-enemy

NOTICE: due to the options of the action method, blobs are not moving horizontally or vertically, this is because Q-learning
learns to use the walls and when tries to move diagonally it moves up and down. 
NOTICE: moving diagonally means that as we initialize the environment most of the times player can't access the food without
moving vertically
"""


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return self.x + self.y  # TODO: check it

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y

    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        if choice == 1:
            self.move(x=-1, y=-1)
        if choice == 2:
            self.move(x=-1, y=1)
        if choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1,
                                        2)  # this means [-1,0,1] (since it can be zero means that this movement can go up or down.
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:  # position > 9 (it can't be 11, since our grid is 10x10)
            self.x = SIZE - 1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1


# this returns the observation by finding the deltas btw player-food, player-enemy
# Initialize the q-table by adding any combination.
if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE + 1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            for x2 in range(-SIZE + 1, SIZE):
                for y2 in range(-SIZE + 1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]  # the key is tuple of
                    # tuples because the observation of the environment includes the deltas btw player-food, player-enemy
                    # this will be like (delta_x_fromFood, delta_y_fromFood), (delta_x_fromEnemy, delta_y_fromEnemy)
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []
for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0:
        print("on #" + str(episode) + ", epsilon:" + str(epsilon))
        print(str(SHOW_EVERY) + "ep mean" + str(np.mean(episode_rewards[-SHOW_EVERY:])))
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):    # Number of steps.
        obs = (player - food, player - enemy)   # get observation
        if np.random.random() > epsilon:    # choose action based on epsilon
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)

        player.action(action)

        # TODO: Maybe later..
        # enemy.move()
        # food.move()

        #   At this point player has already made his move (player.action()) so the environment can determine his reward
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        new_obs = (player - food, player - enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        #   update q-table
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)     # 10x10x3 because its RBG data
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]

            img = Image.fromarray(env, "RGB")
            img = img.resize((300, 300))
            cv2.imshow("", np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")    # ??

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel("rward" + str({SHOW_EVERY}) + "ma")
plt.xlabel("episode #")
plt.show()

with open("blob_checkpoints/qtable-" + str(int(time.time())) + ".pickle", "wb") as f:
    pickle.dump(q_table, f)
