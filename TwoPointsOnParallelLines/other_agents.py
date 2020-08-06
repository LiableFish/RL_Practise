import numpy as np


class SinCosUAgent:
    def __init__(self, env):
        self.u_action_max = env.u_action_max

    def get_action(self, state):
        t, x1, x2, x3, x4 = state
        return self.u_action_max * np.sin(x1) * np.cos(x2) * np.sin(x3) * np.sin(x4) * np.sin(2 * t)


class NormUAgent:
    def __init__(self, env):
        self.u_action_max = env.u_action_max

    def get_action(self, state):
        x = state[1:]
        return - self.u_action_max * np.linalg.norm(x) / (1 + np.linalg.norm(x))


class OptimalConstantCounterVAgent:
    def __init__(self, env, u_agent):
        self.env = env
        self.u_agent = u_agent
        self.beta = self.get_beta()

    def get_beta(self, partition_diameter=0.01):
        betas = np.arange(-self.env.v_action_max, self.env.v_action_max, partition_diameter)
        total_rewards = []

        for beta in betas:
            total_reward = 0
            state = self.env.reset()
            done = False
            while not done:
                u_action = self.u_agent.get_action(state)
                v_action = beta
                state, reward, done, _ = self.env.step(u_action, v_action)
                total_reward += reward
            total_rewards.append(total_reward)

        return betas[np.argmax(total_rewards)]

    def get_action(self, state):
        return self.beta


class ConstantVAgent:
    def __init__(self, env, const=0):
        self.const = const

    def get_action(self, state):
        return self.const


class SinVAgent:
    def __init__(self, env):
        self.v_action_max = env.v_action_max

    def get_action(self, state):
        t, x1, x2, x3, x4 = state
        return self.v_action_max * np.sin(2 * np.pi * t)
