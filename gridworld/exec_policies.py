import os
import argparse
from grid import GridEnv, GridState
from glob import glob
from tqdm import tqdm

MODEL = '../saved_files/10x50000x200x__re(25, -300, -1)__pass4__avg__4.41__success rate__0.97286_episodes_collected2500'
MODEL2 = '../saved_files/10x50000x200x__re(25, -300, -1)__pass4__avg__4.41__success rate__0.97286_episodes_collected1000000'
MODEL3 = os.path.join('..', 'saved_files', 'dizzy40%-Truex50000pass4__avg__-147.17__success rate__0.97826_episodes_collected2500')
FOLDER = 'policies'


def execute_policy(model, policies_folder, policy_nr, dizzy=False, episodes=2500, pr=False, render=False):
    print("Executing\nmodel:{0}\nfolder:{1}\ndizzy:{2}".format(model,policies_folder,dizzy))
    if dizzy:
        env = GridEnv(dizzy=True)
    else:
        env = GridEnv()

    policy_file = os.path.join(model, policies_folder, 'policy_' + str(policy_nr) + '.csv')
    t1 = open(policy_file, 'r')
    policy = t1.readlines()
    t1.close()

    episode_rewards = []
    total_reward = 0
    total_successes = 0

    for episode in range(episodes):
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
                env.reset()
                if reward == env.food_reward:
                    result = "\t\t< SUCCESS! >"
                    total_successes += 1
                elif reward == env.enemy_penalty:
                    result = "\t\t< FAILURE! >"
                else:
                    result = "\t\t< SURVIVED! >"
                if pr:
                    print("|-- Episode {} finished after {} steps.".format(episode, step) + result)

            episode_rewards.append(episode_reward_counter)
        success_rate = total_successes / episodes

    if pr:
        print(
            "\n\nPolicy: {0}\n Total reward collected: {1}\nTotal successes: {2}\nSuccess ratio: {3}".format(policy_nr,
                                                                                                             total_reward,
                                                                                                             total_successes,
                                                                                                             success_rate))
    return total_reward, total_successes, success_rate


def execute_policies(model, policies_folder, exec_max=None, episodes=2500):
    dir = os.path.join(model, policies_folder, '*.csv')
    policies_list = glob(dir)

    if exec_max is not None:
        policies_list = policies_list[:exec_max]

    policy_nr = policies_list.__len__()

    rwd_count = 0
    sc_count = 0
    sr_count = 0

    rwd_max = float('-inf')
    policy_max = None

    env = GridEnv()

    for i, policy in enumerate(tqdm(policies_list)):
        rwd, sc, sr = execute_policy(model, policies_folder, i, env)

        rwd_count += rwd
        sc_count += sc
        sr_count += sr

        if rwd > rwd_max:
            rwd_max = rwd
            policy_max = i

    print("\n\nAverages between {0} policies:".format(policy_nr))
    print("reward {0}".format(rwd_count / policy_nr))
    print("successes {0}".format(sc_count / policy_nr))
    print("success ratio {0}".format(sr_count / policy_nr))
    print("\nBest policy is {0} with {1} reward\n\n".format(policy_max, rwd_max))


# execute_policy(MODEL3,FOLDER,policy_nr=474,pr=True, render=True, dizzy_agent=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="model to execute", required=False)
    parser.add_argument("-mx", help="max policies to execute", default=None)
    parser.add_argument("-p", help="policy to execute", required=False)
    parser.add_argument("-v", help="model version", required=False)
    parser.add_argument("--r", help="render", action="store_true")
    parser.add_argument("--d", help="dizzy agent", action="store_true")

    args = parser.parse_args()
    dizzy = False
    render = False
    model = MODEL
    folder = FOLDER

    if args.v:
        folder = folder + "V" + str(args.v)
    if args.m:
        model = args.m
    if args.r:
        render = True
    if args.d:
        dizzy = True

    execute_policy(model=model, policies_folder=folder, policy_nr=args.p, dizzy=dizzy, render=render)
