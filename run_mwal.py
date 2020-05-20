import os

import numpy as np
from gridworld.expert import process_trajectories
from gridworld.grid import GridEnv
from transition import ThetaEstimator, load_files
from mwal import mwal


def run_mwal(env, expert_file, m=None):
    """
    :param env: The learned environment.
    :param expert_file: Name of the expert file containing trajectories used to teach the apprentice.
    :param m: The number of expert trajectories used to teach the apprentice (for manually insertion).
    :return:
    """
    gamma = 0.95
    T = 500
    weak_estimation = False

    m_req = ThetaEstimator(env).compute_trajectories_threshold()
    if os.path.exists(os.path.join('saved_files', expert_file)):
        F, THETA, E = load_files(expert_file)
        THETA = THETA.todok()
    else:
        if m < m_req:
            weak_estimation = True
        F, THETA, E = process_trajectories(expert_file, env, m, weak_estimation, True)

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
    # #write_out_policy(PP[i, :])


if __name__ == '__main__':
    # environment = GridEnv()
    run_mwal(env=None,
             expert_file='10x50000x200x__re(25, -300, -1)__pass4__avg__4.41__success rate__0.97286_episodes_collected2500', m=2500)
