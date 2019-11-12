import numpy as np
from .base import Arm


class BernoulliPeriodicArm(Arm):
    def __init__(self, p_min, p_max, period, offset=0):
        super(BernoulliPeriodicArm, self).__init__()
        assert 0 < p_min < p_max < 1, "wrong initialisation of probability"
        self.p_min = p_min
        self.p_max = p_max
        self.offset = offset
        self.period = period

    def p_success(self, t):
        y = np.sin(2 * np.pi * (t + self.offset) / self.period)
        return (self.p_max - self.p_min) / 2 * y + (self.p_max + self.p_min) / 2

    def get_expectation(self, t):
        return self.p_success(t)

    def get_reward(self, t):
        return np.random.binomial(1, self.p_success(t))