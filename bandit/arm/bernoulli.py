import numpy as np
from .base import Arm


class BernoulliArm(Arm):
    def __init__(self, p_success):
        super(BernoulliArm, self).__init__()
        self.p_success = p_success

    def get_expectation(self, t):
        return self.p_success

    def get_reward(self, t):
        return np.random.binomial(1, self.p_success)