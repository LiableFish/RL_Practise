import numpy as np
from UnequalGame.CCEM_agent import CCEMAgent
from UnequalGame.unequal_game_env import UnequalGame


def generate_session(u_agent, v_agent, env, t_max=200):
    """
    Generate session on environment with agent
    """
    states = []
    u_actions = []
    v_actions = []
    total_reward = 0
    done = False
    state = env.reset()
    while not done:
        u_action = u_agent.get_action(state)
        v_action = v_agent.get_action(state)
        next_state, reward, done, _ = env.step(u_action[0], v_action[0])
        states.append(state)
        u_actions.append(u_action)
        v_actions.append(v_action)
        total_reward += reward
        state = next_state

    return {'states': states, 'u_actions': u_actions, 'v_actions': v_actions, 'total_reward': total_reward}


def main():
    # env = gym.make('Pendulum-v0')
    # state_shape = env.observation_space.shape
    # action_shape = env.action_space.shape

    env = UnequalGame()
    u_agent = CCEMAgent((2,), (1,), percentile_param=30, action_max=env.u_action_max, reward_param=-1)
    v_agent = CCEMAgent((2,), (1,), percentile_param=70, action_max=env.v_action_max, reward_param=1)

    n_epochs = 100
    n_sessions = 100
    last_mean_reward = 0
    epsilon = 1e-10

    for epoch in range(n_epochs):
        sessions = [generate_session(u_agent, v_agent, env) for _ in range(n_sessions)]
        mean_reward = np.mean([session['total_reward'] for session in sessions])
        u_agent.fit(sessions, prefix='u_')
        v_agent.fit(sessions, prefix='v_')
        print('epoch: {0}, mean reward: {1}'.format(epoch, mean_reward))
        if np.abs(last_mean_reward - mean_reward) < epsilon:
            break
        last_mean_reward = mean_reward


if __name__ == '__main__':
    main()
