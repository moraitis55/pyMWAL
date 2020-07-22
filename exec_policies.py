import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from gridworld.grid import GridEnv, GridState
from glob import glob
from tqdm import tqdm
from statistics import mean, stdev

MODEL = '../saved_files/10x50000x200x__re(25, -300, -1)__pass4__avg__4.41__success rate__0.97286_episodes_collected2500'
MODEL2 = '../saved_files/10x50000x200x__re(25, -300, -1)__pass4__avg__4.41__success rate__0.97286_episodes_collected1000000'
MODEL3 = os.path.join('..', 'saved_files',
                      'dizzy40%-Truex50000pass4__avg__-147.17__success rate__0.97826_episodes_collected2500')
FOLDER = 'policies'


def get_policy_expected_rwd(policy, exp_reward_function, env, env_respawn, episodes=2500):
    print("testing policy for expected reward in {0} episodes".format(episodes))

    total_rwd = 0

    for episode in range(episodes):
        env.reset()

        for step in range(env.episode_steps):
            st = GridState(
                player_position=(env.player.x, env.player.y),
                food_position=(env.food.x, env.food.y),
                enemy_position=(env.enemy.x, env.enemy.y)
            )
            st_index = env.state_space_index[st.__str__()]

            action = int(policy[st_index])
            new_observation, reward, done = env.step(action)
            exp_rwd = exp_reward_function[action, st_index]

            total_rwd += exp_rwd

            if done:
                if env_respawn:
                    env.reset()
                else:
                    break
    return total_rwd


def execute_policy(path_to_policy, policy_nr, env=None, episodes=2500, render=False, render_label='', dizzy=False,
                   pr=False, env_respawn=False, render_wait=200):
    # print("Executing\nmodel:{0}\ndizzy:{1}".format(path_to_policy, dizzy))

    if env is None:
        env = GridEnv(dizzy=dizzy)

    policy_file = os.path.join(path_to_policy, 'policy_' + str(policy_nr) + '.csv')
    t1 = open(policy_file, 'r')
    policy = t1.readlines()
    t1.close()

    episode_rewards = []
    episode_transitions = []

    # for episode in tqdm(range(episodes), disable=pr):
    for episode in range(episodes):
        # print("{0} starting episode {1}".format(render_label, episode))
        env.reset()

        if pr:
            print("====+====+====+====+====+====+====+====+====")
            print("|-- Episode {} starting.".format(episode))

        episode_reward_counter = 0

        for step in range(env.episode_steps):
            st = GridState(
                player_position=(env.player.x, env.player.y),
                food_position=(env.food.x, env.food.y),
                enemy_position=(env.enemy.x, env.enemy.y)
            )
            st_index = env.state_space_index[st.__str__()]

            action = int(policy[st_index])
            new_observation, reward, done = env.step(action)

            ################### RENDER #######################################
            if render:
                if reward == env.food_reward or reward == env.enemy_penalty:
                    env.render(wait=render_wait + 400,
                               label=render_label)  # freeze the image to make it easy for the viewer
                else:
                    env.render(wait=render_wait, label=render_label)
            ##################################################################

            episode_reward_counter += reward

            if pr:
                if (step % 50 == 0 and step > 1) or done:  # print episode total reward every 50 steps
                    print("| [t = {}]\tReward = {:.4f}".format(step, episode_reward_counter))

            if done:

                episode_transitions.append(step)

                if reward == env.food_reward:
                    result = "\t\t< SUCCESS! >"
                elif reward == env.enemy_penalty:
                    result = "\t\t< FAILURE! >"
                else:
                    result = "\t\t< SURVIVED! >"
                if pr:
                    print("|-- Episode {} finished after {} steps.".format(episode, step) + result)

                if env_respawn:
                    env.reset()
                else:
                    break


        episode_rewards.append(episode_reward_counter)

    rewards_avg = mean(episode_rewards)
    rewards_std = stdev(episode_rewards)

    transitions_avg = mean(episode_transitions)
    transitions_std = stdev(episode_transitions)

    if pr:
        print(
            "\n\nPolicy: {0}\n Avg reward collected: {1}\nAvg successes: {2}".format(policy_nr, rewards_avg,
                                                                                     transitions_avg))
    return rewards_avg, rewards_std, transitions_avg, transitions_std


def show_plot(policy_data, policy_std_data, trans_data, trans_std, policy_dir):

    plt.ion()
    plot_dir = os.path.join(policy_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    labels = [str(policy[0]) for policy in enumerate(policy_data)]
    x_pos = np.arange(len(labels))
    CTEs = policy_data
    error = policy_std_data

    # Build the first plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.8, ecolor='black', capsize=10)
    ax.set_ylabel('Average episode reward')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title("Average episode rewards")
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    print('saving first plot')
    plt.savefig(plot_dir + '/' + 'avg_Rewards.png')

    CTEs2 = trans_data
    error2 = trans_std

    # Build the second plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs2, yerr=error2, align='center', alpha=0.8, ecolor='black', capsize=10)
    ax.set_ylabel('Average episode transitions')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title("Average episode transitions")
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    print('saving second plot')
    plt.savefig(plot_dir + '/' + 'avg_Transitions.png')


def execute_policies(path_to_policies, exec_max=None, dizzy=False, env_respawn=False, episodes=2500, env=None,
                     save_plot_label=False):
    dir = os.path.join(path_to_policies, '*.csv')
    policies_list = glob(dir)

    if exec_max is not None:
        policies_list = policies_list[:exec_max]

    policy_nr = policies_list.__len__()

    # a list of policies avg rewards and std used for plot.
    avg_rwds = []
    std_rwds = []
    # a list of policies avg transitions and std
    avg_trans = []
    std_trans = []

    policy_max = None
    rwd_max = float('-inf')

    if env is None:
        env = GridEnv(dizzy=dizzy)

    for i, policy in enumerate(tqdm(policies_list)):
        reward_avg, reward_std, trans_avg, trans_std = execute_policy(path_to_policy=path_to_policies, policy_nr=i, env=env, env_respawn=env_respawn,
                                     episodes=episodes)

        avg_rwds.append(reward_avg)
        std_rwds.append(reward_std)

        avg_trans.append(trans_avg)
        std_trans.append(trans_std)

        if reward_avg > rwd_max:
            rwd_max = reward_avg
            policy_max = i

    # todo: make new plots
    if save_plot_label:
        show_plot(policy_data=avg_rwds, policy_std_data= std_rwds, trans_data=avg_trans, trans_std=std_trans, policy_dir=path_to_policies)

    avg_reward = mean(avg_rwds)
    avg_transitions = mean(avg_trans)

    # write a log
    with open(os.path.join(path_to_policies, 'plots', 'log.txt'), 'a') as out:
        out.write("Averages between {0} policies:\n".format(policy_nr))
        out.write("avg reward {0}\n".format(avg_reward))
        out.write("avg transitions {0}\n".format(avg_transitions))
        out.write("Best policy is {0} with {1} average reward".format(policy_max, rwd_max))

    print("\n\nAverages between {0} policies:".format(policy_nr))
    print("avg reward {0}".format(avg_reward))
    print("avg transitions {0}".format(avg_transitions))
    print("\nBest policy is {0} with {1} average reward\n\n".format(policy_max, rwd_max))


# execute_policy(MODEL3,FOLDER,policy_nr=474,pr=True, render=True, dizzy_agent=True)
# execute_policies(
#     path_to_policies='../saved_files/10x50000x200x__re(25, -300, -1)__pass4__avg__4.41__success rate__0.97286_episodes_collected1000000/policiesV0', env_respawn=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="model to execute")
    parser.add_argument("-mx", help="max policies to execute", default=None, type=int)
    parser.add_argument("-p", help="policy to execute", required=False)
    parser.add_argument("--r", help="render", action="store_true")
    parser.add_argument("--d", help="dizzy agent", action="store_true")
    parser.add_argument("--s", help="save policies plots", action="store_true")

    args = parser.parse_args()

    if args.p is not None:
        execute_policy(model=args.m, policy_nr=args.p, dizzy=args.d, render=args.r)
    else:
        execute_policies(path_to_policies=args.m, exec_max=args.mx, dizzy=args.d, save_plot_label=args.s)
