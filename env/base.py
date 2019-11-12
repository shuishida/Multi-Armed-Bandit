import numpy as np
from collections import OrderedDict


class Environment(object):
    def __init__(self, bandit, agent, delay=0):
        self.bandit = bandit
        self.agent = agent
        self.delay = delay
        self.t = 0
        self.actions = OrderedDict()
        self.rewards = OrderedDict()
        self.regrets = OrderedDict()

    def init(self):
        pass

    def claim_reward(self, action, delay=0):
        reward, regret = self.bandit.get_reward(action)
        self.rewards[self.t + delay] = reward
        self.regrets[self.t + delay] = regret

    def step(self):
        action = self._step()
        reward = self.rewards[self.t] if self.t in self.rewards else None
        regret = self.regrets[self.t] if self.t in self.regrets else None
        if reward is not None:
            self.agent.get_reward(reward, self.t)
        self.t += 1
        self.bandit.step()
        return action, reward, regret

    def _step(self):
        action = self.agent.step(self.t)
        self.claim_reward(action, delay=self.delay)
        return action

    def run(self, n_steps):
        self.reset()
        actions, rewards, _rewards, cum_rewards, cum_rewards_mean, regrets, _regrets, cum_regrets = [], [], [], [], [], [], [], []
        for i in range(n_steps):
            action, reward, regret = self.step()
            actions.append(action)
            rewards.append(reward if reward is not None else 0)
            if reward is not None:
                _rewards.append(reward)
            cum_rewards.append(np.sum(_rewards) if _rewards else 0)
            cum_rewards_mean.append(np.mean(_rewards) if _rewards else 0)
            regrets.append(regret if regret is not None else 0)
            if regret is not None:
                _regrets.append(regret)
            cum_regrets.append(np.sum(_regrets) if _regrets else 0)

        return actions, rewards, cum_rewards, cum_rewards_mean, cum_regrets

    def reset(self):
        self.bandit.reset()
        self.agent.reset()
        self.t = 0
        self.actions = []
        self.init()
