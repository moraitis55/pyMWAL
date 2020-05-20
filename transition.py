import datetime
import os

import numpy as np
from scipy.sparse import dok_matrix, save_npz, load_npz
from tqdm import tqdm
from sklearn.preprocessing import normalize


def save_files(fname, F=None, THETA=None, E=None, PP=None, MM=None):
    print("\nSaving files..")
    dir = os.path.join('saved_files', fname)
    if not os.path.exists('saved_files'):
        os.mkdir('saved_files')
    if not os.path.exists(dir):
        os.mkdir(dir)
    if F is not None:
        with open(os.path.join(dir, 'F.pickle'), 'wb') as f:
            np.save(f, F)
    if E is not None:
        with open(os.path.join(dir, 'E.pickle'), 'wb') as f:
            np.save(f, E)
    if PP is not None:
        with open(os.path.join(dir, 'PP.pickle'), 'wb') as f:
            np.save(f, PP)
    if MM is not None:
        with open(os.path.join(dir, 'MM.pickle'), 'wb') as f:
            np.save(f, MM)
    if THETA is not None:
        save_npz(os.path.join(dir, 'THETA'), THETA.tocsr())
    print("\nSave completed successfully!")


def load_files(fname):
    print("\nLoading files..")
    dir = os.path.join('saved_files', fname)
    with open(os.path.join(dir, 'F.pickle'), 'rb') as f:
        F = np.load(f)
    with open(os.path.join(dir, 'E.pickle'), 'rb') as f:
        E = np.load(f)
    THETA = load_npz(os.path.join(dir, 'THETA.npz'))
    print("\nLoad completed successfully!")
    return F, THETA, E


class ThetaEstimator:
    """
    Constructs an MDP/R Mz, which is subset of the original MDP/R M, with an MLE estimation of the transition function
    by observing the expert's trajectories.

    Attributes:

        K: SzA x Sz matrix, where SzA is the number of state-action pairs, and Sz is the number of states
            including the absorbing state s0. K(i, j) denote the actual number of times (s,a) is visited in the m trajectories.
            (s,a) is (ε, Η)-visited by πΕ if K(s,a) > a lower bound.
            Note: TR is sparse.

        TRz: SzA x Sz normalized version of transition matrix (contains probabilities). TR(i, j) is the probability
            of transitioning to state j under state-action pair i.
            Note: TR_norm is sparse

        Fz: Sz x K feature matrix, where Sz is the number of states including the absorbing state s0, and K is the number
            of features. F(i, j) is the jth feature value for the ith state.
            Note: F(s0, j) = -1

        E: The 1 x K vector of "feature expectations" for the expert's policy. E(i) is the expected cumulative
          discounted value for the ith feature when following the expert's policy (with respect to initial state
          distribution).

        nS: Total number of environment states.

        nSz: Total number of environment states plus the required "absorbing" state s0 to build the transition estimation.

        nA: Total number of actions.

        est_trans_error: Error factor in the estimation of transition from the expert.

        est_trans_delta: Threshold factor of the accuracy in the compute of transition estimation of the expert.

        trans_counter: Counter to inform about the number of observed trajectory steps.
    """
    K = ...  # type: dok_matrix

    def __init__(self, env, m=None, est_trans_error=0.1, est_trans_delta=0.1, weak_estimation=False, fname=None, modify_env=False):

        if modify_env:
            # add one more state in environment (absorbing state)
            env.create_absorbing_state_indexes()

        self.est_trans_error = est_trans_error
        self.est_trans_delta = est_trans_delta
        self.weak_estimation = weak_estimation
        self.nS = env.state_space_size
        self.nA = env.action_space_size
        self.K = dok_matrix((self.nS * self.nA, self.nS))
        self.TRz = None
        if weak_estimation:
            self.TRz = dok_matrix((self.nS * self.nA, self.nS))
            # initialize all action-state sets to absorbing states
            self.K[:, -1] = 1

        self.Fz = np.zeros(shape=(self.nS, env.feature_nr))
        # the absorbing state is initialized with -1 value for all the features.
        self.Fz[self.nS-1,:] = -1

        self.E_counts = np.zeros(shape=(env.feature_nr,))
        self.E = None
        self.m = m
        if m is not None:
            self.total_steps = self.m * env.episode_steps
        self.trans_counter = 0
        self.fname = fname

    def add_transition(self, sa, st_next, st, feats, step):
        """
        Initialize the transition estimation matrix from the expert's visited states.
        :param sa: Current state-action set index.
        :param st_next: The landing state after taking the action.
        :param st: Current state index.
        :param feats: Set of feature values corresponding to current state.
        :param step: Current step in trajectory.
        :return: Updates transition matrix, transition normalized matrix via reference object)
        """
        self.trans_counter += 1

        self.K[sa, st_next] += 1
        if self.weak_estimation:
            self.K[sa, -1] = 0

        self.update_feature_matrix(st, feats)
        self.update_exp_fe(step, feats)

        # if trajectory steps finished, construct the probability distribution.
        if self.trans_counter == self.total_steps:
            if not self.weak_estimation:
                self.construct_theta_estimation()
            if self.fname is not None:
                self.normalize_trans_matrix()
                save_files(self.fname, self.Fz, self.TRz, self.E)

    # def construct_theta_weak_estimation(self):
    #     # for na in tqdm(range(self.nS * self.nA), total=self.nS * self.nA, desc="Constructing theta estimation:"):
    #     #     if self.K[na].sum() == 0:
    #     #         self.K[na, -1] = 1
    #
    #     self.normalize_trans_matrix()

    def construct_theta_estimation(self):
        bound = self.compute_visited_count_threshold()

        for na in tqdm(range(self.nS * self.nA), desc="Constructing theta estimation:"):
            if self.K[na].sum() < bound:
                self.K[na, :self.nS] = 0
                self.K[na, -1] = 1

        self.K[-self.nA:, -1] = 1
        self.normalize_trans_matrix()

    def normalize_trans_matrix(self):
        """
        Normalize the transition matrix to probability distribution
        """
        print("\nNormalizing theta matrix..")
        self.TRz = normalize(self.K, norm='l1', axis=1)
        print("Done")

    def compute_trajectories_threshold(self):
        """
        Compute the number of input trajectories needed from the expert in order to have a MLE for θ
        """
        S = abs(self.nS)
        A = abs(self.nA)
        e = self.est_trans_error
        delta = self.est_trans_delta

        return ((S ** 3 * A) / (8 * e ** 3)) * np.log((S ** 3 * A) / (delta * e)) + S * A * np.log((2 * S * A) / delta)

    def compute_visited_count_threshold(self):
        """
        Compute the threshold of the actual number of times each (s,a) is visited in the m trajectories.
        """
        S = abs(self.nS)
        A = abs(self.nA)
        e = self.est_trans_error

        return (S ** 2 / (4 * e ** 2)) * np.log((S ** 3 * A) / e)

    def update_feature_matrix(self, st, feats):
        """
        This function constructs the feature matrix from the expert's visited states.

        :param st: A state in an expert trajectory.
        :param feats: The set of features corresponding to the specific state.
        """
        for i, feat in enumerate(feats):
            if self.Fz[st][i] != 0 and self.Fz[st][i] != feat:
                raise Exception("ERROR: There was a value conflict while collecting the expert features!")
            self.Fz[st][i] = feat

    def update_exp_fe(self, step, feats):
        """
        Method to estimate expert feature expectations.

        :param step: A state in an expert trajectory.
        :param feats: The set of features corresponding to the specific state.
        """
        gamma = 0.95

        for i, feat in enumerate(feats):
            self.E_counts[i] += gamma ** step * np.array(feat)
        self.E = self.E_counts / self.m
