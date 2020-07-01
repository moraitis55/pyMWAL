import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from gridworld.grid import GridEnv, GridState
from glob import glob
from tqdm import tqdm

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


def execute_policy(path_to_policy, policy_nr, env=None, dizzy=False, episodes=2500, pr=False, render=False, env_respawn=False,
                   return_extra_statistics=False):
    print("Executing\nmodel:{0}\ndizzy:{1}".format(path_to_policy, dizzy))

    if env is None:
        env = GridEnv(dizzy=dizzy)

    policy_file = os.path.join(path_to_policy, 'policy_' + str(policy_nr) + '.csv')
    t1 = open(policy_file, 'r')
    policy = t1.readlines()
    t1.close()

    episode_rewards = []
    total_reward = 0
    total_successes = 0

    episodes_reward_counter = []
    episodes_success_counter = []

    for episode in tqdm(range(episodes)):
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
                render_wait = 50
                if reward == env.food_reward or reward == env.enemy_penalty:
                    env.render(render_wait + 400)  # freeze the image to make it easy for the viewer
                else:
                    env.render(wait=render_wait)
            ##################################################################

            episode_reward_counter += reward
            total_reward += reward

            if pr:
                if (step % 50 == 0 and step > 1) or done:  # print episode total reward every 50 steps
                    print("| [t = {}]\tReward = {:.4f}".format(step, episode_reward_counter))

            if done:
                if reward == env.food_reward:
                    result = "\t\t< SUCCESS! >"
                    total_successes += 1
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
        episodes_success_counter.append(total_successes)

        episodes_reward_counter.append(total_reward)
        success_rate = total_successes / episodes

    if pr:
        print(
            "\n\nPolicy: {0}\n Total reward collected: {1}\nTotal successes: {2}\nSuccess ratio: {3}".format(policy_nr,
                                                                                                             total_reward,
                                                                                                             total_successes,
                                                                                                             success_rate))
    if return_extra_statistics:
        return total_reward, total_successes, success_rate, episodes_reward_counter, episode_rewards, episodes_success_counter
    else:
        return total_reward, total_successes, success_rate


def save_plot(policy_data, policy_dir, label='reward'):
    plot_dir = os.path.join(policy_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    num_policies = policy_data.__len__()
    y_val = [x for x in policy_data]
    x_val = [x[0] for x in enumerate(policy_data)]
    plt.plot(x_val, y_val)
    plt.plot(x_val, y_val, 'or')
    plt.xlabel('policy')
    plt.ylabel('2500 episodes ' + label)
    plt.grid(True)
    plt.show()
    plt.savefig(plot_dir + '/' + label)


def execute_policies(path_to_policies, exec_max=None, dizzy=False, env_respawn=False, episodes=2500, env=None, save_plot=False):
    dir = os.path.join(path_to_policies, '*.csv')
    policies_list = glob(dir)

    if exec_max is not None:
        policies_list = policies_list[:exec_max]

    policy_nr = policies_list.__len__()

    rwd_count = 0
    sc_count = 0
    sr_count = 0

    # a list of policies total rewards used for count plot.
    policies_rwds = []

    rwd_max = float('-inf')
    rwd_min = float('+inf')
    policy_max = None

    if env is None:
        env = GridEnv(dizzy=dizzy)

    for i, policy in enumerate(tqdm(policies_list)):
        rwd, sc, sr = execute_policy(path_to_policy=path_to_policies, policy_nr=i, env=env, env_respawn=env_respawn, episodes=episodes)

        rwd_count += rwd
        sc_count += sc
        sr_count += sr

        policies_rwds.append(rwd)

        if rwd > rwd_max:
            rwd_max = rwd
            policy_max = i
        if rwd < rwd_min:
            rwd_min = rwd

    if save_plot:
        save_plot(policy_data=policies_rwds, policy_dir=path_to_policies)

    print("\n\nAverages between {0} policies:".format(policy_nr))
    print("reward {0}".format(rwd_count / policy_nr))
    print("successes {0}".format(sc_count / policy_nr))
    print("success ratio {0}".format(sr_count / policy_nr))
    print("\nBest policy is {0} with {1} reward\n\n".format(policy_max, rwd_max))


# execute_policy(MODEL3,FOLDER,policy_nr=474,pr=True, render=True, dizzy_agent=True)
# execute_policies(
#     path_to_policies='../saved_files/10x50000x200x__re(25, -300, -1)__pass4__avg__4.41__success rate__0.97286_episodes_collected1000000/policiesV0', env_respawn=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="model to execute")
    parser.add_argument("-mx", help="max policies to execute", default=None)
    parser.add_argument("-p", help="policy to execute", required=False)
    parser.add_argument("--r", help="render", action="store_true")
    parser.add_argument("--d", help="dizzy agent", action="store_true")
    parser.add_argument("--s", help="save policies plots", action="store_true")

    args = parser.parse_args()

    if args.p is not None:
       execute_policy(model=args.m, policy_nr=args.p, dizzy=args.d, render=args.r)
    else:
       execute_policies(path_to_policies=args.m, exec_max=args.mx, dizzy=args.d, save_plot=args.s)
       
