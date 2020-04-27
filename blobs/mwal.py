import ast
import csv
import math
import pickle
import time

import numpy as np
from tqdm import tqdm


class Blob_state:

    def __init__(self, csv_entry):
        self.episode = int(csv_entry[0][0])
        self.step = int(csv_entry[0][1])
        self.player_posx = int(csv_entry[0][2])
        self.player_posy = int(csv_entry[0][3])
        self.food_posx = int(csv_entry[0][4])
        self.food_posy = int(csv_entry[0][5])
        self.enemy_posx = int(csv_entry[0][6])
        self.enemy_posy = int(csv_entry[0][7])
        self.manhattan_food = float(csv_entry[0][8])
        self.manhattan_enemy = float(csv_entry[0][9])
        self.feats = ast.literal_eval(csv_entry[0][10])
        self.action = int(csv_entry[0][11])
        self.reward = float(csv_entry[0][12])


def get_trajectories(model, trajectories_threshold=None, exec_collector=False):
    with open('expert_trajectories/' + model + '.csv', 'r') as f:
        rd = csv.reader(f)
        for row in zip(rd):
            yield row


class mwalAgent:
    """
    This class implements the MWAL algorithm from:
    %
    % Syed, U., Schapire, R. E. (2007) "A Game-Theoretic Approach to Apprenticeship Learning"
    %
    % Here's a description of the parameters:
    %
    % Input:
    %
    % m_theta: The SA x S transition matrix, where SA is the number of state-action pairs, and A is the number of actions.
    %   theta(i, j) is the probability of transitioning to state j under state-action pair i. NB: THETA should be sparse.
    %
    % m_feat: The S x K feature matrix, where S is the number of states, and K is the number of features. F(i, j) is the
    %   jth feature value for the ith state.
    %
    % gamma: The discount factor, which must be a real number in [0, 1).
    %
    % T: The number of iterations to run the algorithm. More iterations yields better results.
    %
    % me: The 1 x K vector of "feature expectations" for the expert's policy. v_expFe(i) is the expected cumulative
    %   discounted value for the ith feature when following the expert's policy (with respect to initial state
    %   distribution).
    %
    % INIT_FLAG: If this is 'first', then initial state distribution is concentrated at state 1. If this is 'uniform',
    %   then initial state distribution is uniform
    %
    % Output:
    %
    % PP: The T x N matrix of output policies. PP(i, j) is the action taken in the jth state
    %   by the ith output policy. To achieve the guarantees of the algorithm, one must, at time 0, choose to follow
    %   exactly one of the output policies, each with probability 1/T.
    %
    % MM: The T x K matrix of output "feature expectations". MM(i, j) is the expected cumulative discounted value for
    %   the jth feature when following the ith output policy (and when starting at the initial state distribution).
    %
    % ITER: A 1 x T vector of value iteration iteration counts. ITER(i) is the number of iterations used by the ith
    %   invocation of value iteration.
    %
    % TT: A 1 x T vector of value iteration running times. TT(i) is the number of seconds used by the ith invocation of
    %   value iteration.
    %
    """

    def __init__(self, model, trajectory_length, k, est_trans_error, est_trans_delta, env_size, episodes=1000,
                 gamma=0.95,
                 action_number=4, m=None):

        # env variables
        self.env_size = env_size
        self.action_number = action_number
        self.state_number = env_size ** 2 - 1  # because states come from manhattan dist between blobs, excluding case
        # more than two blobs in the same position for now
        self.model = model

        # expert variables
        self.H = trajectory_length  # the length of each independent trajectory
        self.k = k  # number of features
        self.m = m
        # self.m = self._compute_trajectories_threshold(est_trans_delta, est_trans_error)  # number of trajectories
        self.K = self._compute_visited_count_threshold(est_trans_error)  # number of times each (s,a) is visited in

        # run variables
        self.episodes = episodes  # training episodes
        self.gamma = gamma  # discount factor

        # Step 1:
        # compute the expert's feature expectations.
        me = self._getEstimateExpFE

        # Step 2:
        # Define const beta.
        beta = (1 + 1 / math.sqrt((2 * math.log(self.k)) / self.episodes))

        # Step 3:
        # Define G(i,μ) game matrix.

        # Step 4:
        # initialize Weights vector
        W = [np.ones(self.k)]

        # Step 5: iteration.
        for i in range(self.episodes):
            # set Weights
            W[i] = [W[i] / W[i].sum()]

            # compute feature matrix
            m_feat = self._compute_feature_matrix()

            # Step 7:
            # compute the reward function based on the expert policy taken from the explored trajectory.
            RE = self._get_estimated_reward_function(m_feat, W[i])

            # compute an ep-optimal policy pt for M with respect to reward function R(s) = wt * fs

            # Step 8:
            # compute an ef-good estimate of feature expectations of the policy found above

            # Step 9:
            # update Weights, trajectory_length
        # end for

        # Step 11:
        # Post-processing step: Return the mixed policy ψ that assigns probability 1/T to π̂(t) , for all t ∈
        # {1, . . . , T }

    def _getEstimateExpFE(self):
        """
        Method to estimate expert feature expectations.
        :return:
        """
        me = np.empty(shape=(self.k,))
        for row in tqdm(get_trajectories(self.model), total=self.m * self.H, desc="Estimating expert's FE:"):
            st = Blob_state(row)
            for i, feat in enumerate(st.feats):
                me[i] += self.gamma ** st.step * np.array(feat)
        return me / self.m

        # for m in range(self.m):
        #     for H in range(self.H):
        #         f1 = (math.pow(self.gamma, H) * np.array(trajectories[m]['feats']["fxd"][H]))
        #         f2 = (math.pow(self.gamma, H) * np.array(trajectories[m]['feats']["fyd"][H]))
        #         f3 = (math.pow(self.gamma, H) * np.array(trajectories[m]['feats']["exd"][H]))
        #         f4 = (math.pow(self.gamma, H) * np.array(trajectories[m]['feats']["eyd"][H]))
        #         me += (f1, f2, f3, f4)
        # me = me / self.m
        # return me

    def _get_estimated_reward_function(self, m_feat, weight_vector):
        """
        Compute the estimated reward function from the features of the expert's trajectories.

        :param m_feat: The feature matrix.
        :param weight_vector: The weight vector of the algorithm.
        :return:
        """
        rf = np.copy(m_feat)
        for i, w in enumerate(weight_vector):
            rf[:, :, :, :, :, :, i] *= w
        return rf

    def _compute_feature_matrix(self):
        m_feat = np.full(
            shape=(self.env_size, self.env_size, self.env_size, self.env_size, self.env_size, self.env_size, self.k),
            fill_value=-1)
        for row in tqdm(get_trajectories(self.model), total=self.m * self.H, desc="Creating feature matrix:"):
            t = Blob_state(row)
            for i, feat in enumerate(t.feats):
                # todo remove after checking
                update_value = \
                    m_feat[t.player_posx][t.player_posy][t.food_posx][t.food_posy][t.enemy_posx][t.enemy_posy][i]
                if update_value != -1:
                    update_value = feat
                else:
                    update_value = feat
        return m_feat

    def _createQtable(self, env_size):
        """
        Creates the Q table.
        Use tuples because the observation of the environment includes the deltas btw player-food, player-enemy
        this will be like (delta_x_fromFood, delta_y_fromFood), (delta_x_fromEnemy, delta_y_fromEnemy).

        :param env_size: environment size
        :return:
        """
        q_table = {}
        for x1 in range(-env_size + 1, env_size):
            for y1 in range(-env_size + 1, env_size):
                for x2 in range(-env_size + 1, env_size):
                    for y2 in range(-env_size + 1, env_size):
                        q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in
                                                         range(4)]  # the key is tuple of
        return q_table

    def _compute_trajectories_threshold(self, delta, e):
        """
        Compute the number of input trajectories needed from the expert in order to have a MLE for θ

        :param delta: probability offset of the expression
        :param e: error percentage
        :return:
        """
        S = abs(self.state_number)
        A = abs(self.action_number)

        return ((S ** 3 * A) / (8 * e ** 3)) * np.log((S ** 3 * A) / (delta * e)) + S * A * np.log((2 * S * A) / delta)

    def _compute_visited_count_threshold(self, e):
        """
        Compute the threshold of the actual number of times each (s,a) is visited in the m trajectories.

        :param e: error percentage
        :return:
        """
        S = abs(self.state_number)
        A = abs(self.action_number)

        return (S ** 2 / (4 * e ** 2)) * np.log((S ** 3 * A) / e)


agent = mwalAgent(model='10x50000x200x__re(25, -300, -1)__pass4__avg__4.41__success rate__0.97286_episodes70000',
                  k=4, trajectory_length=200, est_trans_delta=0.98, est_trans_error=0.9, m=70000, env_size=10)
