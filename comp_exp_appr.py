import os

from gridworld.Q import execute_expert
from gridworld.grid import GridEnv
from exec_policies import execute_policy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse

PLOT_MARKER = ['v', 'o', '+']
PLOT_LS = ['--', '--', '--']
PLOT_C = ['r', 'm', 'c']

PLOT2_C = ['r', 'c', 'y']


def create_statistics(path_to_expert, paths_to_apprectices: list, policies_to_exec: list, policy_labels: list, dizzy,
                      figure_name, iter, episodes):
    env = GridEnv(dizzy=dizzy)
    print("running the expert")
    expert_episode_avg_reward, expert_episode_sum_reward, expert_episode_success = execute_expert(model=path_to_expert,
                                                                                                  dizzy=dizzy, env=env,
                                                                                                  iter=iter,
                                                                                                  episodes=episodes)

    policies_data = []
    policies_data2 = []
    policies_data3 = []
    for j, path in tqdm(enumerate(paths_to_apprectices)):
        print("running apprentice: {0}".format(path))
        policy_data = np.zeros((episodes,))
        policy_data2 = np.zeros((episodes,))
        policy_data3 = np.zeros((episodes,))
        for i in tqdm(range(iter)):
            dump, dump, dump, episodes_reward_counter, episodes_sum_rewards, episodes_success_counter = execute_policy(
                path_to_policy=path,
                policy_nr=policies_to_exec[
                    j],
                env=env,
                return_extra_statistics=True,
                episodes=episodes)
            episodes_reward_counter = np.asarray(episodes_reward_counter)
            episodes_sum_rewards = np.asarray(episodes_sum_rewards)
            episodes_success_counter = np.asarray(episodes_success_counter)
            policy_data = np.add(policy_data, episodes_reward_counter)
            policy_data2 = np.add(policy_data2, episodes_sum_rewards)
            policy_data3 = np.add(policy_data3, episodes_success_counter)
        policy_data_avg = policy_data / iter
        policy_data_avg2 = policy_data2 / iter
        policy_data_avg3 = policy_data3 / iter
        policies_data.append(policy_data_avg)
        policies_data2.append(policy_data_avg2)
        policies_data3.append(policy_data_avg3)

    # expert_episode_avg_reward = np.arange(2500)
    # for i in range(3):
    #     policy_data = np.arange(2500)
    #     policies_data.append(policy_data)

    fig = plt.figure(1)
    x = np.arange(episodes)
    ax = fig.add_subplot(111)
    ax.plot(x, expert_episode_avg_reward, c='m', ls='--', label='expert')
    for i, policy in enumerate(policies_data):
        ax.plot(x, policy, c=PLOT2_C[i], ls=PLOT_LS[i], label=policy_labels[i])
    # ax.plot(x, policies_data[0], c='k', ls='--', label=policy_labels[0])
    ax.set_title('Episode reward sum')
    ax.set_xlabel('episode')
    ax.set_ylabel('reward sum')
    plt.legend(loc='best')
    plt.draw()
    plt.show()

    fig2 = plt.figure(2)
    x = list(range(episodes))
    x = x[0::100]
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, expert_episode_sum_reward[0::100], c='m', ls='--', label='expert')
    for i, policy in enumerate(policies_data2):
        policy = policy[0::100]
        ax2.plot(x, policy, c=PLOT2_C[i], ls=PLOT_LS[i], label=policy_labels[i])
    ax2.set_title('Episode rewards')
    ax2.set_xlabel('episode')
    ax2.set_ylabel('reward')
    plt.legend(loc='best')
    plt.draw()
    plt.show()

    fig3 = plt.figure(3)
    x = np.arange(episodes)
    ax3 = fig3.add_subplot(111)
    ax3.plot(x, expert_episode_success, c='m', ls='--', label='expert')
    for i, policy in enumerate(policies_data3):
        ax3.plot(x, policy, c=PLOT2_C[i], ls=PLOT_LS[i], label=policy_labels[i])
    ax3.set_title('Episode success sum')
    ax3.set_xlabel('episode')
    ax3.set_ylabel('success sum')
    plt.legend(loc='best')
    plt.draw()
    plt.show()

    if not os.path.exists("statistics"):
        os.mkdir("statistics")
    fig.savefig(os.path.join('statistics', 'reward_counter.png'))
    fig2.savefig(os.path.join('statistics', 'sum_rewards.png'))
    fig3.savefig(os.path.join('statistics', 'success_counter.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--expert-model", help="path to expert model")
    parser.add_argument("-a", "--apprentice-models", help="list of paths to apprentice models", nargs='+')
    parser.add_argument('-pn', '--policies-numbers',
                        help='number of policies to execute from the corresponding apprentice'
                             'model', nargs='+', type=int)
    parser.add_argument('-pl', '--policies-labels', help='apprentice policies labels list', nargs='+')
    parser.add_argument('--d', help='To execute in dizzy world', action='store_true')
    parser.add_argument('-fn', help='plot figure name')
    parser.add_argument('-it', help='number of iterations to run the agents', type=int)
    parser.add_argument('-ep', help='number of episodes to run, default 2500', type=int, default=2500)
    args = parser.parse_args()

    create_statistics(path_to_expert=args.expert_model, paths_to_apprectices=args.apprentice_models,
                      policies_to_exec=args.policies_numbers, policy_labels=args.policies_labels,
                      dizzy=args.d, figure_name=args.fn, iter=args.it, episodes=args.ep)
