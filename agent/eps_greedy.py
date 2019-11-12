import numpy as np
from .base import Agent


class EpsilonGreedy(Agent):
    def __init__(self, n_actions, epsilon=None):
        name = r"$\epsilon$-greedy with $\epsilon$={}".format(epsilon) if epsilon is not None else r"Optimal $\epsilon$-greedy algorithm"
        super(EpsilonGreedy, self).__init__(name)
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.count_actions = None
        self.exp_reward = None
        self.init()

    def init(self):
        self.count_actions = np.zeros(self.n_actions)
        self.exp_reward = np.zeros(self.n_actions)

    def _step(self, t):
        if self.epsilon is None:
            epsilon = (t + 1) ** (-1/3) * (self.n_actions * np.log(t + 1)) ** (1/3)
        else:
            epsilon = self.epsilon
        valid_actions = np.arange(self.n_actions)
        if np.random.random() > epsilon:
            r = self.exp_reward
            valid_actions = valid_actions[r == r.max()]
        action = np.random.choice(valid_actions)
        self.count_actions[action] += 1
        return action

    def _get_reward(self, action, reward, t):
        self.exp_reward[action] += (reward - self.exp_reward[action]) / self.count_actions[action]
