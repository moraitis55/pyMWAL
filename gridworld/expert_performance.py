import argparse
from Q import execute_expert2
from grid import GridEnv

def expert_stats(path_to_expert):
    env = GridEnv()
    exp_rwd_avg, exp_rwd_std, exp_trans_avg, exp_trans_std = execute_expert2(model=path_to_expert, env=env)

    print('expert reward average {0}'.format(exp_rwd_avg))
    print('expert transition average {0}'.format(exp_trans_avg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--expert-model", help="path to expert model")
    args = parser.parse_args()
    expert_stats(path_to_expert=args.expert_model)

