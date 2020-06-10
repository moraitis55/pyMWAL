import datetime
import math
import time

import numpy as np
from scipy.sparse import dok_matrix, coo_matrix
from tqdm import tqdm
from opt_policy_and_feat_exp import opt_policy_and_feat_exp
from transition import save_files


def mwal(THETA, F, E, gamma, INIT_FLAG, T=500, tol=None, fname=None, test_env=None):
    """
    This class implements the MWAL algorithm from:

     Syed, U., Schapire, R. E. (2007) "A Game-Theoretic Approach to Apprenticeship Learning"

     Here's a description of the parameters:

     Input:

     # todo: remove them if not wanted.
     K: S x A matrix, denoting the actual number of times (s, a) is visited in the m trajectories.
       For every (s,a), (s,a) is (ε,H)-visited by PE, if K(s,a) exceeding a lower bound.
     K_bound: Lower bound used to define (s,a) as (ε,H)-visited by PE.
     R: reward function, 1xS vector where R(i) is the total reward the agent is awarded for being in the ith state.
     PE: The 1 x N vector that describes expert policy. PE(i) is the action taken in the ith state by the expert policy.
    # todo: end.

     THETA: The SA x S transition matrix, where SA is the number of state-action pairs plus the "dead state", and
       A is the number of actions. THETA(i, j) is the probability of transitioning to state j under state-action pair i.
       NB: THETA should be sparse.

     F: The S x K feature matrix, where S is the number of states, and K is the number of features. F(i, j) is the
       jth feature value for the ith state.

     E: The 1 x K vector of "feature expectations" for the expert's policy. E(i) is the expected cumulative
       discounted value for the ith feature when following the expert's policy (with respect to initial state
       distribution).

     gamma: The discount factor, which must be a real number in [0, 1).

     INIT_FLAG: If this is 'first', then initial state distribution is concentrated at state 1. If this is 'uniform',
       then initial state distribution is uniform

     T: The number of iterations to run the algorithm. More iterations yields better results.

     Output:

     PP: The T x N matrix of output policies_VVzero. PP(i, j) is the action taken in the jth state
       by the ith output policy. To achieve the guarantees of the algorithm, one must, at time 0, choose to follow
       exactly one of the output policies_VVzero, each with probability 1/T.

     MM: The T x K matrix of output "feature expectations". MM(i, j) is the expected cumulative discounted value for
       the jth feature when following the ith output policy (and when starting at the initial state distribution).

     ITER: A 1 x T vector of value iteration iteration counts. ITER(i) is the number of iterations used by the ith
       invocation of value iteration.

     TT: A 1 x T vector of value iteration running times. TT(i) is the number of seconds used by the ith invocation of
       value iteration.

       :type THETA: dok_matrix
       :type F: np.ndarray
    """

    S, K = F.shape
    beta = 1 / (1 + math.sqrt((2 * math.log(K)) / T))

    # Choose initial feature expectations randomly
    VV = np.random.rand(S, K)
    # initialize absorbing state value with zero.
    VV[S - 1] = -1
    VV = coo_matrix(VV)

    # initialize weights list with first weight vector.
    W = np.ones(K)

    RR = []
    ITER = np.ndarray((T,))
    TT = np.ndarray((T,))
    MM = np.ndarray((T, K))
    PP = np.ndarray((T, S))

    for i in tqdm(range(T), desc="MWAL episode:"):
        # set Weights
        w = W / W.sum()
        t1 = time.time()


        P, M, VV, ITER[i], exp_rwd = opt_policy_and_feat_exp(THETA=THETA, F=F, GAMMA=gamma, w=w, INIT_FLAG=INIT_FLAG, VV=VV, tol=tol, test_env=test_env)

        RR.append(exp_rwd)

        # terminal condition (case where expert is dominated)
        featdiff = M - E
        if featdiff.min() > 0:

            if i == 0:
                TT[1] = time.time() - t1
            else:
                TT[i] = TT[i - 1] + time.time() - t1

            PP[i, :] = P
            MM[i, :] = M
            break

        G = ((1 - gamma) * (M - E) + 2 * np.ones((1, K))) / 4
        W = W * beta ** G
        W = W.squeeze(axis=0)

        if i == 0:
            TT[i] = time.time() - t1
        else:
            TT[i] = TT[i - 1] + time.time() - t1

        PP[i, :] = P
        MM[i, :] = M

    # end for

    if fname is not None:
        save_files(fname, PP=PP, MM=MM)

    return PP, MM, ITER, TT, RR
