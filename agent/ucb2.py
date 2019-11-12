import numpy as np
from .base import Agent


class UCB2(Agent):
    def __init__(self, n_actions, alpha):
        super(UCB2, self).__init__(r"UCB2 where $\alpha$={}".format(alpha))
        self.n_actions = n_actions
        self.alpha = alpha
        self.count_actions = None
        self.exp_reward = None
        self.count_r = None
        self.selected_action = None
        self.init()

    def init(self):
        self.count_actions = np.zeros(self.n_actions)
        self.exp_reward = np.zeros(self.n_actions)
        self.count_r = np.zeros(self.n_actions)
        self.selected_action = None

    def tau(self, r):
        return np.ceil((1.0 + self.alpha) ** r)

    def ucb2(self):
        tau = self.tau(self.count_r)
        radius = np.sqrt((1 + self.alpha) * np.log(np.e * self.count_actions.sum() / tau) / (2 * tau))
        return self.exp_reward + radius

    def _step(self, t):
        if self.count_actions.min() == 0:
            action = self.count_actions.argmin()
        else:
            while True:
                if self.selected_action is None:
                    action = self.ucb2().argmax()
                    r_a = self.count_r[action]
                    n_times = self.tau(r_a + 1) - self.tau(r_a)
                    self.selected_action = (action, n_times)
                action, n_remaining = self.selected_action
                if n_remaining == 0:
                    self.selected_action = None
                    self.count_r[action] += 1
                else:
                    break
            self.selected_action = (action, n_remaining - 1)
        self.count_actions[action] += 1
        return action

    def _get_reward(self, action, reward, t):
        self.exp_reward[action] += (reward - self.exp_reward[action]) / self.count_actions[action]
