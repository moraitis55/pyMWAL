import csv
import os

from gridworld.expert import process_trajectories
from gridworld.grid import GridEnv
from transition import ThetaEstimator, load_files
from write_policy import write_out_policies
from mwal import mwal
from exec_policies import execute_policies, show_plot

def count_file(file):
    print("Reading expert file..")
    filepath = os.path.join('gridworld', 'expert_trajectories', file)
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        lines = len(list(reader))
    return lines

def run_mwal(env, expert_file, dizzy=False, env_respawn=False, m=None):
    """
    :param env: The learned environment.
    :param expert_file: Name of the expert file containing trajectories used to teach the apprentice.
    :param dizzy: tells the environment if the agent is dizzy (stochastic environment)
    :param m: The number of expert trajectories used to teach the apprentice (for manually insertion).
    :return:
    """
    gamma = 0.95
    T = 500
    weak_estimation = False

    # Number of trajectory steps.
    total_steps = count_file(expert_file)

    if os.path.exists(os.path.join('saved_files', expert_file)):
        F, THETA, E = load_files(expert_file)
        THETA = THETA.todok()
    else:
        m_req = ThetaEstimator(env, total_steps).compute_trajectories_threshold()
        if m < m_req:
            weak_estimation = True
        F, THETA, E = process_trajectories(expert_file, env, m, weak_estimation, total_steps, True)

    env = GridEnv(dizzy=dizzy)
    # Run the mwal algorithm
    PP, MM, ITER, TT, RR = mwal(THETA=THETA, F=F, E=E, gamma=gamma, INIT_FLAG='first', T=T, fname=expert_file,
                                test_env=env)

    # # Determine the mixing coefficients (trivial)
    # c = np.ones((T, 1)) / T
    #
    # # Choose a policy at random according to the mixing coefficients.
    # C = np.zeros((T, 1))
    # C[0] = c[0]
    # for i in range(T):
    #     C[i] = C[i - 1] + c[i]
    #
    # r = np.random.rand()
    # i = np.argwhere(r <= C)[0][0]
    #
    # # Write out that policy

    dir = write_out_policies(PP, expert_file)
    execute_policies(path_to_policies=dir, dizzy=dizzy, env_respawn=env_respawn, env=env, save_plot=True)
    show_plot(policy_data=RR, label='expected reward', policy_dir=dir)



file1='episodic_dizzy40%_False_50000pass4__avg__4.16__success_rate__0.97136_episodes_collected90000.csv'
file2='episodic_dizzy40%_True_50000pass4__avg__-38.5__success_rate__0.9005_episodes_collected1000000'

if __name__ == '__main__':
    environment = GridEnv(dizzy=True)
    run_mwal(env=environment ,
             expert_file=file2,
             m=2500,
             dizzy=True)
