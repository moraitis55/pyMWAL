import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from blobs.blob import Blob
from blobs.blob_environment import BlobEnv
from matplotlib import style

style.use("ggplot")


class BlobQAgent:
    def __init__(self, loadQtable: str = None, blob_env_size=10, steps=200, episodes=25000, show_every=3000,
                 epsilon=0.9, epsilon_decay=0.9998, lr=0.1, discount=0.95, checkpoint_name=None):
        self.loadQ = loadQtable
        self.steps = steps
        self.learning_rate = lr
        self.discount = discount
        self.episodes = episodes
        self.Q = self._createQtable()
        self.blov_env_size = blob_env_size
        self.show_every = show_every
        self.epsilon = epsilon
        self.epsDecay = epsilon_decay
        self.checkpoint_name = checkpoint_name

    def _createQtable(self):
        if self.loadQ is None:
            q_table = {}
            for x1 in range(-self.blov_env_size + 1, self.blov_env_size):
                for y1 in range(-self.blov_env_size + 1, self.blov_env_size):
                    for x2 in range(-self.blov_env_size + 1, self.blov_env_size):
                        for y2 in range(-self.blov_env_size + 1, self.blov_env_size):
                            q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in
                                                             range(4)]  # the key is tuple of
                            # tuples because the observation of the environment includes the deltas btw player-food, player-enemy
                            # this will be like (delta_x_fromFood, delta_y_fromFood), (delta_x_fromEnemy, delta_y_fromEnemy)
        else:
            with open(self.loadQ, "rb") as f:
                q_table = pickle.load(f)
        return q_table

    def run(self):

        env = BlobEnv()

        episode_rewards = []
        for episode in range(self.episodes):

            if episode % self.show_every == 0:
                print("on #" + str(episode) + ", epsilon:" + str(self.epsilon))
                print(str(self.show_every) + "ep mean" + str(np.mean(episode_rewards[-self.show_every:])))
                show = True
            else:
                show = False

            episode_reward = 0
            for i in range(self.steps):
                obs = env.reset()

                if np.random.random() > self.epsilon:  # choose action based on epsilon
                    action = np.argmax(self.Q[obs])
                else:
                    action = np.random.randint(0, env.action_space_size)  # 4 if vertical movement not allowed

                new_observation, reward, done = env.step(action)

                max_future_q = np.max(self.Q[new_observation])
                current_q = self.Q[obs][action]

                #   update q-table
                if reward == env.food_reward:
                    new_q = env.food_reward
                elif reward == env.enemy_penalty:
                    new_q = env.enemy_penalty
                else:
                    new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
                                reward + self.discount * max_future_q)

                self.Q[obs][action] = new_q  # update Q value

                if show:
                    env.render()

                episode_reward += reward
                if reward == env.food_reward or reward == env.enemy_penalty:
                    break

            episode_rewards.append(episode_reward)
            self.epsilon *= self.epsDecay

        moving_avg = np.convolve(episode_rewards, np.ones((self.show_every,)) / self.show_every, mode="valid")

        plt.plot([i for i in range(len(moving_avg))], moving_avg)
        plt.ylabel("rward" + str({self.show_every}) + "ma")
        plt.xlabel("episode #")
        plt.show()

        if self.checkpoint_name:
            with open("blob_Q_checkpoints/"+ self.checkpoint_name + "-" + str(int(time.time())) + ".pickle", "wb") as f:
                pickle.dump(self.Q, f)
