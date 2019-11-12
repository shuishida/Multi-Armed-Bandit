import numpy as np
from .base import Agent


class SuccessiveElimination(Agent):
    def __init__(self, n_actions, n_steps):
        super(SuccessiveElimination, self).__init__("Successive elimination")
        self.n_steps = n_steps
        self.n_actions = n_actions
        self.count_actions = None
        self.exp_reward = None
        self.active_actions = None
        self.round_actions = None
        self.init()

    def init(self):
        self.count_actions = np.zeros(self.n_actions)
        self.exp_reward = np.zeros(self.n_actions)
        self.active_actions = np.ones(self.n_actions)
        self.round_actions = list(np.arange(self.n_actions))

    def calc_conf_radius(self):
        return np.sqrt(2 * np.log(self.n_steps) / self.count_actions)

    def ucb(self):
        return self.exp_reward + self.calc_conf_radius()

    def lcb(self):
        return self.exp_reward - self.calc_conf_radius()

    def _step(self, t):
        if self.round_actions:
            action = self.round_actions.pop()
        elif self.active_actions.sum() == 1:
            action = self.active_actions.argmax()
        else:
            lcb, ucb = self.lcb(), self.ucb()
            lcb_max = lcb[self.active_actions.astype(bool)].max()
            stay_active = ucb >= lcb_max
            self.active_actions *= stay_active
            self.round_actions = list(np.arange(self.n_actions)[self.active_actions.astype(bool)])
            action = self.round_actions.pop()
        self.count_actions[action] += 1
        return action

    def _get_reward(self, action, reward, t):
        self.exp_reward[action] += (reward - self.exp_reward[action]) / self.count_actions[action]
