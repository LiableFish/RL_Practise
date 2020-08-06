import numpy as np


class TwoPointsOnParallelLines:
    def __init__(self, initial_x=np.array([4.5, 0, 0, 0]), alpha=1, m1=0.01, m2=1, theta=10,
                 u_action_max=2.5, v_action_max=1, dt=0.005):
        self.initial_x = initial_x
        self.alpha = alpha
        self.m1 = m1
        self.m2 = m2
        self.theta = theta
        self.u_action_max = u_action_max
        self.v_action_max = v_action_max
        self.dt = dt
        self.state = self.reset()

    def reset(self):
        self.state = np.hstack((0, self.initial_x))
        return self.state

    def step(self, u_action, v_action):
        t, x1, x2, x3, x4 = self.state
        x1 = x1 + x2 * self.dt
        x2 = x2 + (- (self.alpha / self.m1) * x2 + (1 / self.m1) * u_action) * self.dt
        x3 = x3 + x4 * self.dt
        x4 = x4 + (1 / self.m2) * v_action * self.dt
        t += self.dt
        self.state = np.array([t, x1, x2, x3, x4])

        reward = 0
        done = False
        if t >= self.theta:
            reward = - np.abs(x1 - x3)
            done = True

        return self.state, reward, done, None
