import numpy as np


class UnequalGame:
    """Unequal game environment implementation for RL practise"""

    def __init__(self, initial_x=1, dt=0.005, terminal_time=2, u_action_max=2, v_action_max=1):
        """Create an environment
        :param initial_x: starting point for u_agent
        :param dt: time rate
        :param terminal_time: stopping time
        :param u_action_max: maximum action value for u_agent
        :param v_action_max: maximum action value for v_agent
        """
        self.u_action_max = u_action_max
        self.v_action_max = v_action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.initial_x = initial_x
        self.state = self.reset()

    def reset(self):
        """Reset environment state for a new game session
        :return: starting state
        """
        self.state = np.array([0, self.initial_x])
        return self.state

    def step(self, u_action, v_action):
        """Generate a new environment state under the actions of agents
        :param u_action: u_agent action
        :param v_action: v_agent action
        :return: new state, reward, done flag, None
        """
        t, x = self.state
        x = x + (u_action - v_action) * self.dt
        t += self.dt
        self.state = np.array([t, x])

        reward = 0
        done = False
        if t >= self.terminal_time:
            reward = x ** 2
            done = True

        return self.state, reward, done, None
