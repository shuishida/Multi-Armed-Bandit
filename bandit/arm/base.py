class Arm(object):
    def __init__(self):
        pass

    def get_expectation(self, t):
        raise NotImplementedError

    def get_reward(self, t):
        raise NotImplementedError
