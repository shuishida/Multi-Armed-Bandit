class Bandit(object):
    def __init__(self, n_actions):
        self.t = 0
        self.n_actions = n_actions

    def init(self):
        pass

    def step(self):
        self.t += 1
        self._step()

    def _step(self):
        pass

    def best_expectation(self):
        raise NotImplementedError

    def get_reward(self, action):
        raise NotImplementedError

    def reset(self):
        self.t = 0
        self.init()