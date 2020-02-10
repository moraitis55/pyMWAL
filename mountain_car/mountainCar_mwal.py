import math


class MWAL_Agent:

    def __init__(self, k, episodes, gamma, W):
        self.episodes = episodes  # training episodes
        self.gamma = gamma  # discount factor
        self.W = W  # weight factor
        self.k = k  # number of features
        self.beta = (1 + 1 / math.sqrt((2 * math.log(self.k)) / self.episodes))
        self.G


