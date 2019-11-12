import numpy as np
from .base import Agent


class ExploreExploit(Agent):
    def __init__(self, n_actions, n_explore):
        super(ExploreExploit, self).__init__("Explore and exploit algorithm $N={}$".format(n_explore))
        self.n_explore = n_explore
        self.n_actions = n_actions
        self.count_actions = None
        self.sum_reward = None
        self.chosen_action = None
        self.init()

    def init(self):
        self.count_actions = np.zeros(self.n_actions, dtype=np.int)
        self.sum_reward = np.zeros(self.n_actions)

    def _step(self, t):
        count = self.count_actions.sum()
        if self.count_actions.min() < self.n_explore:
            action = self.count_actions.argmin()
        elif count == self.n_explore * self.n_actions:
            action = np.random.choice(np.arange(self.n_actions)[self.sum_reward == self.sum_reward.max()])
            self.chosen_action = action
        else:
            action = self.chosen_action
        self.count_actions[action] += 1
        return action

    def _get_reward(self, action, reward, t):
        self.sum_reward[action] += reward


class ExploreExploitOptimal(ExploreExploit):
    def __init__(self, n_actions, n_steps):
        n_explore = int((n_steps / n_actions * np.sqrt(2 * np.log(n_steps))) ** (2/3))
        super(ExploreExploitOptimal, self).__init__(n_actions, n_explore)
        self.name = r"Optimal explore and exploit algorithm ($N = {}$)".format(n_explore)
