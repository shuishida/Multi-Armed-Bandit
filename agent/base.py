import numpy as np
from collections import OrderedDict


class Agent(object):
    def __init__(self, name):
        self.name = name
        self.history = OrderedDict()

    def init(self):
        pass

    def step(self, t):
        action = self._step(t)
        self.history[t] = (action, np.nan)
        return action

    def _step(self, t):
        raise NotImplementedError

    def get_reward(self, reward, t):
        assert t in self.history, "time t when the action was taken doesn't exist in history"
        action = self.history[t][0]
        self.history[t] = (action, reward)
        self._get_reward(action, reward, t)

    def _get_reward(self, action, reward, t):
        raise NotImplementedError

    def reset(self):
        self.history = OrderedDict()
        self.init()