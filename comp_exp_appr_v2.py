import argparse
import os

from gridworld.Q import execute_expert2
from gridworld.grid import GridEnv
from exec_policies import execute_policy, show_plot


def create_statistics(path_to_expert, paths_to_apprectices: list, policies_to_exec: list, policy_labels: list,
                      save_dir):
    env = GridEnv()
    print("running the expert")
    exp_rwd_avg, exp_rwd_std, exp_trans_avg, exp_trans_std = execute_expert2(model=path_to_expert, env=env)

    pol_rwd_avg = list()
    pol_rwd_std = list()
    pol_tr_avg = list()
    pol_tr_std = list()

    for j, path in enumerate(paths_to_apprectices):
        print("running apprentice: {0}".format(path))

        rewards_avg, rewards_std, transition_avg, transition_std = execute_policy(path, policies_to_exec[j], env)

        pol_rwd_avg.append(rewards_avg)
        pol_rwd_std.append(rewards_std)
        pol_tr_avg.append(transition_avg)
        pol_tr_std.append(transition_std)

    pol_rwd_avg.append(exp_rwd_avg)
    pol_rwd_std.append(exp_rwd_std)
    pol_tr_avg.append(exp_trans_avg)
    pol_tr_std.append(exp_trans_std)

    show_plot(policy_data=pol_rwd_avg, policy_std_data=pol_rwd_std, trans_data=pol_tr_avg, trans_std=pol_tr_std,
              imported_labels=policy_labels, plot_dir=save_dir)

    # write a log
    with open(os.path.join(save_dir, 'log.txt'), 'a') as out:
        for j, label in enumerate(policy_labels):
            out.write("{0} avg reward: {1}\n".format(label, pol_rwd_avg[j]))
        out.write("\n\n")
        for j, label in enumerate(policy_labels):
            out.write("{0} avg transitions: {1}\n".format(label, pol_tr_avg[j]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--expert-model", help="path to expert model")
    parser.add_argument("-a", "--apprentice-models", help="list of paths to apprentice models", nargs='+')
    parser.add_argument('-pn', '--policies-numbers',
                        help='number of policies to execute from the corresponding apprentice'
                             'model', nargs='+', type=int)
    parser.add_argument('-pl', '--policies-labels', help='apprentice policies labels list', nargs='+')
    parser.add_argument('-p', '--path', help='the path to save plots')

    args = parser.parse_args()
    create_statistics(path_to_expert=args.expert_model, paths_to_apprectices=args.apprentice_models,
                      policies_to_exec=args.policies_numbers, policy_labels=args.policies_labels, save_dir=args.path)
