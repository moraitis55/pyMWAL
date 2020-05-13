def estimate_exp_fe(k, model, m, H, gamma):
    """
    Method to estimate expert feature expectations.
    :return:
    """
    me = np.empty(shape=(k,))
    for row in tqdm(trajectory_generator(model), total=m * H, desc="Estimating expert's FE:"):
        st = csvdata_to_trajectory_state(row)
        for i, feat in enumerate(st.feats):
            me[i] += gamma ** st.step * np.array(feat)
    return me / m

    # for m in range(self.m):
    #     for H in range(self.H):
    #         f1 = (math.pow(self.gamma, H) * np.array(trajectories[m]['feats']["fxd"][H]))
    #         f2 = (math.pow(self.gamma, H) * np.array(trajectories[m]['feats']["fyd"][H]))
    #         f3 = (math.pow(self.gamma, H) * np.array(trajectories[m]['feats']["exd"][H]))
    #         f4 = (math.pow(self.gamma, H) * np.array(trajectories[m]['feats']["eyd"][H]))
    #         me += (f1, f2, f3, f4)
    # me = me / self.m
    # return me

def estimate_reward_function(F, weight_vector, nS):
    """
    Compute the estimated reward function from the features of the expert's trajectories.

    :param F: S x K feature matrix, where S is the number of states, and K is the number of features. F(i, j) is the
    jth feature value for the ith state.
    :param weight_vector: The weight vector of the algorithm.
    :param nS: State space
    :return: R: 1xS vector where R(i) is the total reward the agent is awarded for being in the ith state.
    """
    F_copy = np.copy(F)
    R = np.empty(shape=nS)
    for i, w in enumerate(weight_vector):
        F_copy[:, i] *= w
    for s in tqdm(range(nS), desc="Constructing the reward function"):
        R[s] = F_copy[s].sum()
    return R

def construct_expert_policy(step, PE: np.array, env):
    """
    This function constructs the expert's policy vector.
    :param step: A step in an expert trajectory.
    :param PE: 1 x S vector that describes expert policy. PE(i) is the action taken in the ith state by the expert policy.
    :param env: The environment.
    :return: Updates expert's policy vector via reference object.
    """
    t = csvdata_to_trajectory_state(step)
    # map state values to state index
    state_index = env.state_space_index[t.state_before.__str__()]
    PE[state_index] = t.action
