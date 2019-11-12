import numpy as np
from .base import Arm


class NormalArm(Arm):
    def __init__(self, mean, var):
        super(NormalArm, self).__init__()
        self.mean = mean
        self.var = var

    def get_expectation(self, t):
        return self.mean

    def get_reward(self, t):
        return np.random.normal(self.mean, self.var)