import numpy as np
from UnequalGame.CCEM_agent import CCEMAgent
from UnequalGame.unequal_game_env import UnequalGame


def generate_session(u_agent, v_agent, env, test=False):
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
        u_action = u_agent.get_action(state, test=test)
        v_action = v_agent.get_action(state)
        actions = (u_action[0], v_action[0]) if str(u_agent) == 'u_' else (v_action[0], u_action[0])
        next_state, reward, done, _ = env.step(*actions)
        states.append(state)
        u_actions.append(u_action)
        v_actions.append(v_action)
        total_reward += reward
        state = next_state

    return {'states': states, f'{u_agent}actions': u_actions, f'{v_agent}actions': v_actions,
            'total_reward': total_reward}


def fit_epoch(u_agent, v_agent, env, n_sessions, test):
    sessions = [generate_session(u_agent, v_agent, env, test=test) for _ in range(n_sessions)]
    mean_reward = np.mean([session['total_reward'] for session in sessions])
    if not test:
        u_agent.fit(sessions)
    v_agent.fit(sessions)
    return mean_reward


def fit_agents(u_agent, v_agent, env, n_epochs, n_sessions, epsilon, test=False):
    last_mean_reward = 0
    mean_rewards = []

    for epoch in range(n_epochs):
        mean_reward = fit_epoch(u_agent, v_agent, env, n_sessions, test)
        mean_rewards.append(mean_reward)
        print(f'epoch: {epoch}, mean reward: {mean_reward}')
        if np.abs(last_mean_reward - mean_reward) < epsilon:
            break
        last_mean_reward = mean_reward

    return u_agent, np.array(mean_rewards)


def fit_agents_one_by_one(u_agent, v_agent, env, n_epochs, n_sessions, n_iter_for_fit, epsilon):
    last_mean_reward = 0
    mean_rewards = []
    fit_agent = u_agent
    wait_agent = v_agent
    epoch = 0
    stop = False

    while not stop and epoch < n_epochs:

        for _ in range(n_iter_for_fit):
            mean_reward = fit_epoch(wait_agent, fit_agent, env, n_sessions, test=True)
            mean_rewards.append(mean_reward)
            print(f'epoch: {epoch}, current agent: {fit_agent}, mean reward: {mean_reward}')
            if np.abs(last_mean_reward - mean_reward) < epsilon:
                stop = True
                break
            last_mean_reward = mean_reward
            epoch += 1

        print('\n')
        wait_agent, fit_agent = fit_agent, wait_agent

    return u_agent, np.array(mean_rewards)


def main():
    # env = gym.make('Pendulum-v0')
    # state_shape = env.observation_space.shape
    # action_shape = env.action_space.shape

    env = UnequalGame()
    u_agent = CCEMAgent((2,), (1,), percentile_param=30, action_max=env.u_action_max, reward_param=-1)
    v_agent = CCEMAgent((2,), (1,), percentile_param=70, action_max=env.v_action_max, reward_param=1)

    u_agent_one_by_one = CCEMAgent((2,), (1,), percentile_param=30, action_max=env.u_action_max, reward_param=-1)
    v_agent_one_by_one = CCEMAgent((2,), (1,), percentile_param=70, action_max=env.v_action_max, reward_param=1)

    u_fit_agent_one_by_one, mean_rewards_one_by_one = \
        fit_agents_one_by_one(u_agent_one_by_one, v_agent_one_by_one, env,
                              n_epochs=10, n_sessions=5, n_iter_for_fit=2, epsilon=1e-6)


if __name__ == '__main__':
    main()
