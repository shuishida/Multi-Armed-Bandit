import numpy as np
from .base import Agent


class UCB1(Agent):
    def __init__(self, n_actions, n_steps=None):
        super(UCB1, self).__init__("UCB1" if n_steps is None else "UCB1 with fixed total #steps")
        self.n_steps = n_steps
        self.n_actions = n_actions
        self.count_actions = None
        self.exp_reward = None
        self.round_actions = None
        self.init()

    def init(self):
        self.count_actions = np.zeros(self.n_actions)
        self.exp_reward = np.zeros(self.n_actions)
        self.round_actions = list(np.arange(self.n_actions))

    def calc_conf_radius(self):
        T = self.count_actions.sum() if self.n_steps is None else self.n_steps
        return np.sqrt(2 * np.log(T) / self.count_actions)

    def ucb(self):
        return self.exp_reward + self.calc_conf_radius()

    def _step(self, t):
        if self.round_actions:
            action = self.round_actions.pop()
        else:
            action = self.ucb().argmax()
        self.count_actions[action] += 1
        return action

    def _get_reward(self, action, reward, t):
        self.exp_reward[action] += (reward - self.exp_reward[action]) / self.count_actions[action]
