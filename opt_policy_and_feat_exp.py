import numpy as np
from scipy.sparse import coo_matrix, dok_matrix
from exec_policies import get_policy_expected_rwd
from random import randrange


def opt_policy_and_feat_exp(THETA, F, GAMMA, w, INIT_FLAG, VV, test_env, tol=None):
    """

    This function computes the optimal policy with respect to a particular reward function, and simultaneously computes
      the "feature expectations" for that policy.

     Here's a description of the parameters:

    Input:

    THETA: The SA x S transition matrix, where SA is the number of state-action pairs, and A is the number of actions.
      THETA(i, j) is the probability of transitioning to state j under state-action pair i. NB: THETA should be sparse.

    F: The S x K feature matrix, where S is the number of states, and K is the number of features. F(i, j) is the jth
      feature value for the ith state.

    GAMMA: The discount factor, which must be a real number in [0, 1).

    w: A K x 1 vector that is a convex combination, i.e. the components of w are non negative and sum to 1.

    INIT_FLAG: Specifies initial state distribution. Either 'uniform' or 'first' (i.e. concentrated at first state)

    VV: Initial per-state feature expectations. Intended to be carried over from previous invocations.

    Output:

    P: The 1 x S vector that describes the optimal policy with respect to the reward function R = F*w. P(i) is the action
      taken in the ith state by the optimal policy.

    M: The 1 x K vector of "feature expectations" for the optimal policy. M(i) is the expected cumulative discounted value
      for the ith feature when following the optimal policy (and when starting at state 1).

    VV: Final per-state feature expectations. Intended to carry over to future invocations.

    :type VV: np.ndarray
    :type F: np.ndarray
    :type THETA: dok_matrix
    """
    if tol is None:
        tol = 0.0001

    SA, S = THETA.shape
    A = int(SA / S)
    if INIT_FLAG is 'first':
        init = np.zeros((1, S))
        init[0, 0] = 1
    elif INIT_FLAG is 'uniform':
        init = np.ones(1, S) / S

    dummy, K = F.shape
    F_long = np.zeros(shape=(S * A, K))

    # F_long: SA x K feature Q matrix, where F_long(i,j) contains the value of the feature j in the corresponding to the ith pair
    # of state-action
    for i in range(K):
        F_long[:, i] = np.reshape(a=np.kron(np.ones(shape=(1, A)), F[:, i]).conj().T,
                                  newshape=(S * A,))

    # V: S x 1 reward vector, where V(i,j) contains the weighted value of the ith feature expectation in the jth state (Reward).
    V = VV.dot(w).conj().T

    # Conserve memory
    # del F #todo uncommnent later
    F_long = coo_matrix(F_long)
    w = coo_matrix(w)

    delta = tol + 1

    ITER = 0

    exception_number = 0

    while delta > tol:
        print("\nValue Iteration \nstep {0}   delta: {1}".format(str(ITER), str(delta)))
        # value iteration (todo: validate F_long usage as the R(s) of the type??)
        Q = F_long.tocsr() + GAMMA * THETA.tocsr() @ VV.tocsr()
        # SA x 1 sparse vector containing all the values in the SA different combinations (dot product of Q with w gives the reward).
        dummy = Q @ w.T
        # A x S sparse matrix containing values of action-state couples
        QA = dummy.reshape(A, S)
        # 1 x S sparse vector of feature expectations in each state. #todo: validate max, argmax results
        V_new = QA.max(axis=0)
        # 1 x S sparse vector of optimal actions in each state.
        P = np.ndarray(shape=(S,))
        QA = QA.toarray()
        for i in range(S):
            dump = QA[:,i]
            P[i] = np.random.choice(np.where(dump == dump.max())[0])
            #try:
            #    P[i] = np.random.choice(np.where(dump == dump.max())[0])
            #except Exception:
            #    P[i] = randrange(4)
            #    exception_number += 1
        P = P.astype(int)

        # P = QA.toarray().argmax(axis=0)

        # conserve memory
        del dummy
        # del QA

        # AA(i,j,:) = 1 if the ith action was the optimal in state jth, else AA(i,j,:)=0
        AA = np.zeros(shape=(A, S, K))
        for a in range(A):
            AA[a, np.where(P == a), :] = 1  # todo: validate where() func in sparse matrix

        Q = np.reshape(Q.toarray(), (A, S, K))
        VV = np.squeeze(np.sum(a=AA * Q, axis=0))

        # Conserve memory
        VV = coo_matrix(VV)

        delta = np.max(abs(V - V_new))

        V = V_new.toarray()

        ITER = ITER + 1

    exp_rwd = None
    if test_env is not None:
        exp_rwd = get_policy_expected_rwd(policy=P, exp_reward_function=QA, env=test_env, env_respawn=False)

    M = init @ VV.toarray()
    return P, M, VV, ITER, exp_rwd, exception_number
