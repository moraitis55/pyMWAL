import csv
import os

from gridworld.expert import process_trajectories
from gridworld.grid import GridEnv
from transition import ThetaEstimator, load_files
from write_policy import write_out_policies
from mwal import mwal

def count_file(file):
    filepath = os.path.join('gridworld', 'expert_trajectories', file)
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        lines = len(list(reader))
    return lines

def run_mwal(env, expert_file, m=None):
    """
    :param env: The learned environment.
    :param expert_file: Name of the expert file containing trajectories used to teach the apprentice.
    :param m: The number of expert trajectories used to teach the apprentice (for manually insertion).
    :return:
    """
    gamma = 0.95
    T = 150
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

    # Run the mwal algorithm
    PP, MM, ITER, TT = mwal(THETA=THETA, F=F, E=E, gamma=gamma, INIT_FLAG='first', T=T, fname=expert_file)

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

    write_out_policies(PP, expert_file)


if __name__ == '__main__':
    # environment = GridEnv()
    run_mwal(env=None,
             expert_file='episodic_dizzy(0)_50000pass1__avg__-46.0__success rate__0.8485_episodes_collected2500.csv',
             m=2500)
