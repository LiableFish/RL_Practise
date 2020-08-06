import numpy as np


class OptimalUAgent:
    def __init__(self, env):
        super().__init__()
        self.theta = env.theta
        self.m1 = env.m1
        self.alpha = env.alpha
        self.u_action_max = env.u_action_max

    def get_r(self, state):
        t, x1, x2, x3, x4 = state
        nu = self.m1 / self.alpha
        p1 = x1 - nu * (np.exp(- (self.theta - t) / nu) - 1) * x2
        p3 = x3 + (self.theta - t) * x4
        return p1 - p3

    def get_action(self, state):
        r = self.get_r(state)
        if r >= 0:
            return self.u_action_max
        else:
            return -self.u_action_max


class OptimalVAgent:
    def __init__(self, env):
        super().__init__()
        self.theta = env.theta
        self.m1 = env.m1
        self.alpha = env.alpha
        self.v_action_max = env.v_action_max

    def get_r(self, state):
        t, x1, x2, x3, x4 = state
        nu = self.m1 / self.alpha
        p1 = x1 - nu * (np.exp(- (self.theta - t) / nu) - 1) * x2
        p3 = x3 + (self.theta - t) * x4
        return p1 - p3

    def get_action(self, state):
        r = self.get_r(state)
        if r >= 0:
            return self.v_action_max
        else:
            return -self.v_action_max
