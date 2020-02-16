import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from blobs.blob_environment import BlobEnv
from matplotlib import style
import pandas as pd

style.use("ggplot")


class BlobQAgent:
    def __init__(self, loadQtable: str = None, blob_env_size=10, steps=200, episodes=25000, stats_every=3000,
                 enable_render=True, render_every=3000, epsilon=0.9, epsilon_decay=0.9998, lr=0.1, discount=0.95,
                 checkpoint_name=None, silence=False):
        self.loadQ = loadQtable
        self.t = steps
        self.learning_rate = lr
        self.discount = discount
        self.episodes = episodes
        self.blob_env_size = blob_env_size
        self.stats_every = stats_every
        self.enable_render = enable_render
        self.render_every = render_every
        self.epsilon = epsilon
        self.epsDecay = epsilon_decay
        self.checkpoint_name = checkpoint_name
        self._silent = silence

        self.Q = self._createQtable()

    def _createQtable(self):
        if self.loadQ is None:
            q_table = {}
            for x1 in range(-self.blob_env_size + 1, self.blob_env_size):
                for y1 in range(-self.blob_env_size + 1, self.blob_env_size):
                    for x2 in range(-self.blob_env_size + 1, self.blob_env_size):
                        for y2 in range(-self.blob_env_size + 1, self.blob_env_size):
                            q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in
                                                             range(4)]  # the key is tuple of
                            # tuples because the observation of the environment includes the deltas btw player-food, player-enemy
                            # this will be like (delta_x_fromFood, delta_y_fromFood), (delta_x_fromEnemy, delta_y_fromEnemy)
        else:
            with open(self.loadQ, "rb") as f:
                q_table = pickle.load(f)
        return q_table

    def _printif(self, string):
        if not self._silent: print(string)

    def _printWindowStats(self, episode_rewards, episode):
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("\t\t MOVING WINDOW STATISTICS ")
        print("%%%%%%%%%%  EPISODES {0}-{1}  %%%%%%%%%%%%".format(str(episode - self.stats_every),
                                                                  str(episode)))
        print("\tCurrent epsilon value: {0}".format(str(self.epsilon)))
        print("\tMean reward: {0}".format(str(np.mean(episode_rewards[-self.stats_every]))))
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    def run(self):

        env = BlobEnv(return_images=False, size=self.blob_env_size, episode_steps=self.t)

        episode_rewards = []
        for episode in range(self.episodes):

            if episode % self.stats_every == 0 and episode > 0:
                self._printWindowStats(episode_rewards, episode)

            self._printif("====+====+====+====+====+====+====+====+====")
            self._printif("|-- Episode {} starting.".format(episode))

            obs = env.reset()
            total_reward = 0
            for step in range(self.t):

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

                ################### RENDER #######################################
                if self.enable_render and episode % self.render_every == 0:
                    if reward == env.food_reward or reward == env.enemy_penalty:
                        env.render(wait=800)  # freeze the image to make it easy for the viewer
                    else:
                        env.render(250)
                ##################################################################

                total_reward += reward

                if (step % 50 == 0 and step > 1) or done:  # print episode total reward every 50 steps
                    self._printif("| [t = {}]\tReward = {:.4f}".format(step, total_reward))

                if done:
                    if reward == env.food_reward:
                        result = "\t\t< SUCCESS! >"
                    elif reward == env.enemy_penalty:
                        result = "\t\t< FAILURE! >"
                    else:
                        result = "\t\t< SURVIVED! >"
                    self._printif("|-- Episode {} finished after {} steps.".format(episode, step) + result)
                    break

            episode_rewards.append(total_reward)
            self.epsilon *= self.epsDecay

        moving_avg = np.convolve(episode_rewards, np.ones((self.stats_every,)) / self.stats_every, mode="valid")

        plt.plot([i for i in range(len(moving_avg))], moving_avg)
        plt.ylabel("Reward in {0} episodes window".format(str(self.stats_every)))
        plt.xlabel("Episode")
        plt.show()

        if self.checkpoint_name:
            with open("Q_checkpoints/" + self.checkpoint_name + "-" + str(int(time.time())) + ".pickle",
                      "wb") as f:
                pickle.dump(self.Q, f)


agent = BlobQAgent(checkpoint_name='pass1', blob_env_size=10, enable_render=False, episodes=25000, stats_every=3000, silence=True)
agent.run()
