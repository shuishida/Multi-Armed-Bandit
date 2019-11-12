import numpy as np
from .base import Bandit


class IndependentBandit(Bandit):
    def __init__(self, arms):
        super(IndependentBandit, self).__init__(len(arms))
        self.arms = arms

    def best_expectation(self):
        return np.max([arm.get_expectation(self.t) for arm in self.arms])

    def get_reward(self, action):
        arm = self.arms[action]
        return arm.get_reward(self.t), self.best_expectation() - arm.get_expectation(self.t)
