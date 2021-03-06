import copy
import csv
import math
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from gridworld.grid import GridEnv, GridState
from matplotlib import style
from tqdm import tqdm
from statistics import mean, stdev
import pandas as pd
from tqdm import tqdm

style.use("ggplot")


class BlobQAgent:

    def __init__(self, observe_mode=False, loadQtable=None, blob_env_size=10, steps=200, episodes=25000,
                 stats_every=3000, enable_render=True, render_every=None, render_wait=250, render_label='image', epsilon=0.9,
                 epsilon_decay=0.9998, lr=0.1, discount=0.95, checkpoint_name=None, silence=False, fr=25, ep=-300,
                 mp=-1, env_respawn=False, dizzy_agent=False, inspect=False, env=None):
        self.observe = observe_mode
        self.loadQ = loadQtable
        self.t = steps
        self.learning_rate = lr
        self.discount = discount
        self.episodes = episodes
        self.render_every = render_every
        self.render_wait = render_wait
        self.render_label = render_label
        self._silent = silence
        self.inspect = inspect

        # environment variables
        self.blob_env_size = blob_env_size
        self.food_reward = fr
        self.enemy_penalty = ep
        self.move_penalty = mp
        self.env_respawn = env_respawn
        self.dizzy = dizzy_agent
        self.env = env

        if observe_mode:  # in observe mode we don't need exploration
            self.epsilon = 0
            self.epsDecay = 0
            self.checkpoint_name = None
            self.stats_every = None
            self.enable_render = False
            self.df = pd.DataFrame(
                columns=['episode', 'step', 'state_player', 'state_food', 'state_enemy',
                         'euclid_food',
                         'euclid_enemy', 'feats', 'action', 'reward'])
            self.append_list = []
        else:
            self.epsilon = epsilon
            self.epsDecay = epsilon_decay
            self.checkpoint_name = checkpoint_name
            self.stats_every = stats_every
            self.enable_render = enable_render

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
            with open(os.path.join("Q_checkpoints", self.loadQ + ".pickle"), "rb") as f:
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

    def _collect_trajectory_data(self, state_before, state_next, step, episode, action, reward, obs, env, done):
        # euclidean_dist_food = math.sqrt(pow(obs[0][0], 2) + pow(obs[0][1], 2))
        # euclidean_dist_enemy = math.sqrt(pow(obs[1][0], 2) + pow(obs[1][1], 2))

        # manhattan distances
        dfe = abs(state_before.fp.x - state_before.ep.x) + abs(state_before.fp.y - state_before.ep.y)  # food-enemy
        dpf = abs(obs[0][0]) + abs(obs[0][1])  # player -food
        dpe = abs(obs[1][0]) + abs(obs[1][1])  # player -enemy

        # value filters used for feature engineering
        f1 = lambda x: 1 / (x + 1)
        f2 = lambda x: x / (env.size - 1)

        # when distance from food is getting smaller we want the feature to have better price.
        # also when the distance from the enemy is getting bigger we want the feature to have better price.
        feats = [f1(abs(obs[0][0])), f1(abs(obs[0][1])), f2(abs(obs[1][0])), f2(abs(obs[1][1])), f1(dfe), f2(dfe)]

        # todo: added for debuging, remove later.
        if state_before.__str__() in self.debug_states.keys():
            self.debug_states[state_before.__str__()] = [feats, obs, state_before.__str__()]
        else:
            self.debug_states[state_before.__str__()] = [feats, obs, state_before.__str__()]

        # add euclidean_dist + 1 in order to avoid dividing by zero
        self.append_list.append([episode, step, state_before.pp.x, state_before.pp.y, state_before.fp.x,
                                 state_before.fp.y, state_before.ep.x, state_before.ep.y, state_next.pp.x,
                                 state_next.pp.y, state_next.fp.x, state_next.fp.y, state_next.ep.x, state_next.ep.y,
                                 dpf, dpe, feats, action, reward / -env.enemy_penalty])
        if self.env_respawn:
            end_condition = (episode > 0 and episode % 30000 == 0 and step == self.t - 1) or (
                    episode == self.episodes - 1 and step == self.t - 1)
        else:
            end_condition = (episode > 0 and episode % 30000 == 0 and step == self.t - 1) or (
                    episode == self.episodes - 1 and done)
        # append trajectories if it is the last step of the last episode or every 50k episodes.
        if end_condition:
            out_file = os.path.join("expert_trajectories",
                                    self.loadQ + "_episodes_collected" + str(self.episodes) + ".csv")
            if not os.path.exists(out_file):
                with open(out_file, 'w') as f:
                    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                    wr.writerows(self.append_list)
                    self.append_list = []
            else:
                with open(out_file, 'a') as f:
                    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                    wr.writerows(self.append_list)
                    self.append_list = []

    def run(self):

        global trajectory_collection
        if self.observe:
            trajectory_collection = {}

        if self.env is None:
            self.env = GridEnv(return_images=False, size=self.blob_env_size, episode_steps=self.t,
                          move_penalty=self.move_penalty, enemy_penalty=self.enemy_penalty,
                          food_reward=self.food_reward,
                          dizzy=self.dizzy)

        episode_rewards = []
        episode_transitions = []
        # episode_successes = []
        # episode_successes_counter = []
        # success_counter = 0

        episode_reward_counter = []
        reward_counter = 0

        # todo: remove later.
        self.debug_states = {}

        for episode in tqdm(range(self.episodes), desc="Episode", disable=self._silent):
            self.env.reset()

            if self.stats_every and episode % self.stats_every == 0 and episode > 0:
                self._printWindowStats(episode_rewards, episode)

            self._printif("====+====+====+====+====+====+====+====+====")
            self._printif("|-- Episode {} starting.".format(episode))

            total_reward = 0
            total_success = 0
            for step in range(self.t):
                obs = (self.env.player - self.env.food, self.env.player - self.env.enemy)
                state_before = GridState(player_position=copy.copy(self.env.player), food_position=copy.copy(self.env.food),
                                         enemy_position=copy.copy(self.env.enemy))

                if np.random.random() > self.epsilon:  # choose action based on epsilon
                    action = np.argmax(self.Q[obs])
                else:
                    action = np.random.randint(0, self.env.action_space_size)  # 4 if vertical movement not allowed

                new_observation, reward, done = self.env.step(action)

                if self.observe:
                    state_next = GridState(player_position=copy.copy(self.env.player), food_position=copy.copy(self.env.food),
                                           enemy_position=copy.copy(self.env.enemy))
                    self._collect_trajectory_data(state_before, state_next, step, episode, action, reward, obs, self.env,
                                                  done)

                max_future_q = np.max(self.Q[new_observation])
                current_q = self.Q[obs][action]

                #   update q-table
                if reward == self.env.food_reward:
                    new_q = self.env.food_reward
                elif reward == self.env.enemy_penalty:
                    new_q = self.env.enemy_penalty
                else:
                    new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
                            reward + self.discount * max_future_q)

                if not self.inspect:
                    self.Q[obs][action] = new_q  # update Q value

                ################### RENDER #######################################
                if self.render_every and episode % self.render_every == 0:
                    if reward == self.env.food_reward or reward == self.env.enemy_penalty:
                        self.env.render(wait=self.render_wait + 400, label=self.render_label)  # freeze the image to make it easy for the viewer
                    else:
                        self.env.render(wait=self.render_wait, label=self.render_label)
                ##################################################################

                total_reward += reward
                reward_counter += reward

                if (step % 50 == 0 and step > 1) or done:  # print episode total reward every 50 steps
                    self._printif("| [t = {}]\tReward = {:.4f}".format(step, total_reward))

                if done:
                    episode_transitions.append(step)
                    if reward == self.env.food_reward:
                        result = "\t\t< SUCCESS! >"
                        total_success += 1
                    elif reward == self.env.enemy_penalty:
                        result = "\t\t< FAILURE! >"
                    else:
                        result = "\t\t< SURVIVED! >"
                    self._printif("|-- Episode {} finished after {} steps.".format(episode, step) + result)
                    # if in respawn mode reset the environment and start over for the remaining steps of the episode.
                    if self.env_respawn:
                        self.env.reset()
                    else:
                        break

            # if total_success > 0:
            #     success_counter += total_success
            # episode_successes_counter.append(success_counter)

            episode_rewards.append(total_reward)
            # episode_successes.append(total_success)
            # episode_reward_counter.append(reward_counter)
            self.epsilon *= self.epsDecay

        # self.model_avg_reward = round(mean(episode_rewards), 2)
        # self.model_success_rate = success_counter / self.episodes

        rewards_avg = mean(episode_rewards)
        rewards_std = stdev(episode_rewards)

        transitions_avg = mean(episode_transitions)
        transitions_std = stdev(episode_transitions)

        fig_out_name = None
        if self.checkpoint_name:
            # env_string = "{0}x{1}x{2}x__re({3}, {4}, {5})__".format(self.blob_env_size, self.episodes, self.t,
            #                                                         self.food_reward, self.enemy_penalty,
            #                                                         self.move_penalty)
            env_string = "episodic_dizzy(0)_{1}".format(self.dizzy, self.episodes)
            # env_string = "dizzy40%-{0}x{1}".format(self.dizzy, self.episodes)
            out_name = "Q_checkpoints/" + env_string + self.checkpoint_name + "__avg__" + str(
                self.model_avg_reward) + "__success rate__" + str(self.model_success_rate)
            fig_out_name = "Q_checkpoints/figures/" + env_string + self.checkpoint_name + "__avg__" + str(
                self.model_avg_reward) + "__success rate__" + str(self.model_success_rate)
            with open(out_name + ".pickle", "wb") as f:
                pickle.dump(self.Q, f)

        # if self.stats_every:
        #
        #     moving_avg_reward = np.convolve(episode_rewards, np.ones((self.stats_every,)) / self.stats_every,
        #                                     mode="valid")
        #     plt.plot([i for i in range(len(moving_avg_reward))], moving_avg_reward)
        #     plt.ylabel("Average reward in {0} episodes window".format(str(self.stats_every)))
        #     plt.xlabel("Episode")
        #     if fig_out_name:  # save figure
        #         name = fig_out_name + "__rewardChart" + ".png"
        #         plt.savefig(name, bbox_inches='tight')
        #     plt.show()
        #
        #     moving_avg_success = np.convolve(episode_successes, np.ones((self.stats_every,)) / self.stats_every,
        #                                      mode="valid")
        #     plt.plot([i for i in range(len(moving_avg_success))], moving_avg_success)
        #     plt.ylabel("Average successes in {0} episodes window".format(str(self.stats_every)))
        #     plt.xlabel("Episode")
        #     if fig_out_name:  # save figure
        #         name = fig_out_name + "__successChart" + '.png'
        #         plt.savefig(name, bbox_inches='tight')
        #     plt.show()

        # print("Total rewards in {0} episodes: {1}\nTotal successes:{2}".format(self.episodes, sum(episode_rewards),
        #                                                                        sum(episode_successes)))
        return rewards_avg, rewards_std, transitions_avg, transitions_std


def collect_trajectories(nr=6, dizzy=False, checkpoint='10x50000x200x__re(25, -300, -1)__pass4__avg__4.41__success rate__0.97286'):
    start = time.time()
    agent = BlobQAgent(observe_mode=True, loadQtable=checkpoint, episodes=nr, dizzy_agent=dizzy, silence=True)
    agent.run()
    end = time.time()
    print("NOTICE: {0} trajectories were collected in {1} secs / {2} mins".format(nr, end - start, (end - start) / 60))


def inspect_model(model, dizzy=False, render_wait=250, render_label='expert', env=None):
    agent = BlobQAgent(loadQtable=model, render_wait=render_wait, render_every=1, episodes=100000, epsilon=0,
                       env_respawn=True, dizzy_agent=dizzy, observe_mode=True, render_label=render_label, env=env, silence=True)
    agent.run()

def execute_expert2(model, env=None):
    agent = BlobQAgent(loadQtable=model, silence=True, epsilon=0, stats_every=None, inspect=True, env=env, episodes=2500)

    rewards_avg, rewards_std, trans_avg, trans_std = agent.run()
    return rewards_avg, rewards_std, trans_avg, trans_std


def execute_expert(model, dizzy, env_respawn=False, env=None, iter=10, episodes=2500):

    agent = BlobQAgent(loadQtable=model, dizzy_agent=dizzy, env_respawn=env_respawn, silence=True, epsilon=0,
                       episodes=episodes,
                       stats_every=None, inspect=True, env=env)

    episode_reward = np.zeros((episodes,))
    episode_sum_reward = np.zeros((episodes,))
    episode_success = np.zeros((episodes,))
    for i in tqdm(range(iter)):
        episode_sum_rewards, dump, episode_reward_counter, episode_successes_counter = agent.run()
        episode_successes_counter = np.asarray(episode_successes_counter)
        episode_success = np.add(episode_success, episode_successes_counter)

        episode_sum_rewards = np.asarray(episode_sum_rewards)
        episode_sum_reward = np.add(episode_sum_reward, episode_sum_rewards)

        episode_reward_counter = np.asarray(episode_reward_counter)
        episode_reward = np.add(episode_reward, episode_reward_counter)
    episode_avg_reward = episode_reward / iter
    episode_sum_avg_reward = episode_sum_reward / iter
    episode_success_avg = episode_success / iter
    # print("Total rewards in 2500 episodes (average 10 runs): {0}\nTotal successes:{1}".format(reward, success))
    return episode_avg_reward.tolist(), episode_sum_avg_reward.tolist(), episode_success_avg.tolist()

# agent = BlobQAgent(checkpoint_name="pass1", episodes=50000)
# agent = BlobQAgent(checkpoint_name="pass3", episodes=50000, dizzy_agent=True, loadQtable="dizzy-Truex50000pass2__avg__-114.63__success rate__0.9562")
# agent = BlobQAgent(checkpoint_name="pass4", episodes=50000, dizzy_agent=True, loadQtable="dizzy40%-Truex50000pass3__avg__-152.89__success rate__0.97656")
# agent.run()
# collect_trajectories(nr=10, checkpoint='episodic_dizzy40%_True_50000pass4__avg__-38.5__success_rate__0.9005', dizzy=True)
# inspect_model(model="10x50000x200x__re(25, -300, -1)__pass4__avg__4.41__success rate__0.97286", render_wait=50)
# inspect_model(model="dizzy40%-Truex50000pass4__avg__-147.17__success rate__0.97826", render_wait=50, dizzy=True)
# execute_expert('episodic_dizzy40%_False_50000pass4__avg__4.16__success_rate__0.97136', dizzy=False)
