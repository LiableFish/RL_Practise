import numpy as np
from torch.multiprocessing import cpu_count, Pool
from UnequalGame.CCEM_agent import CCEMAgent
from UnequalGame.unequal_game_env import UnequalGame


def generate_session(
    first_agent,
    second_agent,
    env,
    n_iter_update,
    test=False,
):
    """Generate session on environment with agents
    :param first_agent: first agent (u_agent by default)
    :param second_agent: second agent (v_agent by default)
    :param env: environment
    :param n_iter_update: number of iterations between getting new actions
    :param test: if True first agent will not add noise in get_action method
    :return: session dict wit states, first agent actions, second agent actions and total rewards
    """
    states = []
    first_agent_actions = []
    second_agent_actions = []
    total_reward = 0
    done = False
    state = env.reset()
    iteration = 0

    while not done:
        if not iteration % n_iter_update:
            first_agent_action = first_agent.get_action(state, test=test)
            second_agent_action = second_agent.get_action(state)
            states.append(state)
            first_agent_actions.append(first_agent_action)
            second_agent_actions.append(second_agent_action)

        actions = (first_agent_action[0], second_agent_action[0])
        if str(first_agent) == 'v_':
            actions = (actions[1], actions[0])

        next_state, reward, done, _ = env.step(*actions)
        total_reward += reward
        state = next_state
        iteration += 1

    return {
        'states': states,
        f'{first_agent}actions': first_agent_actions,
        f'{second_agent}actions': second_agent_actions,
        'total_reward': total_reward,
    }


class Trainer:
    """Class to fit agents with different ways"""

    def __init__(
        self,
        u_agent,
        env,
    ):
        """Create new network
        :param u_agent: agent that wants to minimize reward
        :param env: environment
        """
        self.u_agent = u_agent
        self.env = env

    def _test_agent(
        self,
        u_agent,
        n_epochs,
        n_sessions,
        n_iter_update,
        epsilon=-1,
    ):
        """Test u_agent by fit a new v_agent
        :param u_agent: agent that wants to minimize reward
        :param n_epochs: number of epochs to fit
        :param n_sessions: number of sessions for one epoch
        :param n_iter_update: number of iterations between getting new actions
        :param epsilon: early stopping criterion (-1 to use all epochs)
        :return: test total rewards
        """
        v_agent = CCEMAgent(
            u_agent.network.input_shape,
            u_agent.network.output_shape,
            percentile_param=100-u_agent.percentile_param,
            action_max=self.env.v_action_max,
            reward_param=1,
        )
        _, rewards, _ = self._fit_agents(
            u_agent,
            v_agent,
            n_epochs,
            n_sessions,
            n_iter_update,
            epsilon,
            test=True,
        )
        return rewards

    def test_agent(
        self,
        n_epochs,
        n_sessions,
        n_iter_update,
        epsilon=-1,
    ):
        """Test u_agent by fit a new v_agent
        :param n_epochs: number of epochs to fit
        :param n_sessions: number of sessions for one epoch
        :param n_iter_update: number of iterations between getting new actions
        :param epsilon: early stopping criterion (-1 to use all epochs)
        :return: test total rewards
        """
        return self._test_agent(
            self.u_agent,
            n_epochs,
            n_sessions,
            n_iter_update,
            epsilon,
        )

    def _fit_epoch(
        self,
        u_agent,
        v_agent,
        n_sessions,
        n_iter_update,
        test,
    ):
        """Fit agents during an one epoch
        :param u_agent: agent that wants to minimize reward
        :param v_agent: agent that wants to maximize reward
        :param n_sessions: number of sessions
        :param n_iter_update: number of iterations between getting new actions
        :param test: if True u_agent will not be fitted
        :return: mean total reward over sessions
        """
        # sessions = [generate_session(self.u_agent, self.v_agent, self.env, test=test) for _ in range(n_sessions)]
        with Pool(cpu_count()) as pool:
            sessions = pool.starmap(
                generate_session,
                [(u_agent, v_agent, self.env, n_iter_update, test) for _ in range(n_sessions)],
            )

        mean_reward = np.mean([session['total_reward'] for session in sessions])

        if not test:
            u_agent.fit(sessions)

        v_agent.fit(sessions)

        return mean_reward

    def _fit_agents(
        self,
        u_agent,
        v_agent,
        n_epochs,
        n_sessions,
        n_iter_update,
        n_test_epochs=300,
        epsilon=-1,
        n_iter_debug=0,
        test=False,
    ):
        """Fit both agent together during several epochs
        :param u_agent: agent that wants to minimze reward
        :param v_agent: agent that wants to maximize reward
        :param n_epochs: number of epochs to fit
        :param n_sessions: number of sessions for one epoch
        :param n_iter_update: number of iterations between getting new actions
        :param n_test_epochs: number of epochs to test agent
        :param epsilon: early stopping criterion (-1 to use all epochs)
        :param n_iter_debug: number of iteration between tests
        :param test: if True u_agent will not be fitted
        :return: u_agent, mean total rewards, test total rewards
        """
        last_mean_reward = 0
        mean_rewards = []
        test_rewards = []

        for epoch in range(n_epochs):

            mean_reward = self._fit_epoch(
                u_agent=u_agent,
                v_agent=v_agent,
                n_sessions=n_sessions,
                n_iter_update=n_iter_update,
                test=test,
            )
            mean_rewards.append(mean_reward)
            print(f'epoch: {epoch}, mean reward: {mean_reward}')
            if np.abs(last_mean_reward - mean_reward) < epsilon:
                break
            last_mean_reward = mean_reward

            if n_iter_debug and (epoch + 1) % n_iter_debug == 0:
                print('\n{:-^50}\n'.format('TEST BEGIN'))
                test_rewards.append(
                    self._test_agent(
                        u_agent=u_agent,
                        n_epochs=n_test_epochs,
                        n_sessions=n_sessions,
                        n_iter_update=n_iter_update,
                        epsilon=epsilon,
                    ),
                )
                print('\n{:-^50}\n'.format('TEST END'))

        return u_agent, np.array(mean_rewards), np.array(test_rewards)

    def fit_agents(
        self,
        v_agent,
        n_epochs,
        n_sessions,
        n_iter_update,
        n_test_epochs=300,
        epsilon=-1,
        n_iter_debug=0,
        test=False,
    ):
        """Fit both agent together during several epochs
        :param v_agent: agent that wants to maximize reward
        :param n_epochs: number of epochs to fit
        :param n_sessions: number of sessions for one epoch
        :param n_iter_update: number of iterations between getting new actions
        :param n_test_epochs: number of epochs to test agent
        :param epsilon: early stopping criterion (-1 to use all epochs)
        :param n_iter_debug: number of iteration between tests
        :param test: if True u_agent will not be fitted
        :return: u_agent, mean total rewards, test total rewards
        """
        return self._fit_agents(
            self.u_agent,
            v_agent,
            n_epochs,
            n_sessions,
            n_iter_update,
            n_test_epochs,
            epsilon,
            n_iter_debug,
            test,
        )

    def fit_agents_one_by_one(
        self,
        v_agent,
        n_epochs,
        n_sessions,
        n_iter_update,
        n_iter_for_fit,
        n_test_epochs=300,
        epsilon=-1,
        n_iter_debug=0,
    ):
        """Fit agents ony by one during several epochs.
        During fix number of iterations one agent will be fitted while the other will not.
        :param v_agent: agent that wants to maximize reward
        :param n_epochs: number of epochs to fit
        :param n_sessions: number of sessions for one epoch
        :param n_iter_update: number of iterations between getting new actions
        :param n_iter_for_fit: number of iterations between agent switching
        :param n_test_epochs: number of epochs to test agent
        :param epsilon: early stopping criterion (-1 to use all epochs)
        :param n_iter_debug: number of iteration between tests
        :return: u_agent, mean  total rewards, test total rewards
        """
        last_mean_reward = 0
        mean_rewards = []
        test_rewards = []
        fitting_agent = self.u_agent
        awaiting_agent = v_agent
        epoch = 0
        stop = False

        while not stop and epoch < n_epochs:

            for _ in range(n_iter_for_fit):

                mean_reward = self._fit_epoch(
                    awaiting_agent,
                    fitting_agent,
                    n_sessions=n_sessions,
                    n_iter_update=n_iter_update,
                    test=True,
                )

                mean_rewards.append(mean_reward)
                print(f'epoch: {epoch}, current agent: {fitting_agent}, mean reward: {mean_reward}')
                if np.abs(last_mean_reward - mean_reward) < epsilon:
                    stop = True
                    break
                last_mean_reward = mean_reward

                if n_iter_debug and (epoch + 1) % n_iter_debug == 0:
                    print('\n{:-^50}\n'.format('TEST BEGIN'))
                    test_rewards.append(
                        self._test_agent(
                            u_agent=self.u_agent,
                            n_epochs=n_test_epochs,
                            n_sessions=n_sessions,
                            n_iter_update=n_iter_update,
                            epsilon=epsilon,
                        ),
                    )
                    print('\n{:-^50}\n'.format('TEST END'))

                epoch += 1
                if epoch >= n_epochs:
                    break

            print('\n')
            awaiting_agent, fitting_agent = fitting_agent, awaiting_agent

        return self.u_agent, np.array(mean_rewards), np.array(test_rewards)

    def fit_random_agent_pairs(
        self,
        u_agents,
        v_agents,
        n_pairs,
        n_epochs,
        n_sessions,
        n_iter_update,
        n_test_epochs=300,
    ):
        """ Fit random pairs of u_ and v_agents
        :param u_agents: agents that wants to minimize reward
        :param v_agents: agents that wants to maximize reward
        :param n_pairs: number of pairs to fit
        :param n_epochs: number of epochs for one pair fit
        :param n_sessions: number of sessions for one epoch
        :param n_iter_update: number of iterations between getting new actions
        :param n_test_epochs: number of epochs to test agent
        :param n_iter_debug: number of iteration between tests
        :return: u_agent that will have minimum test total reward, mean total rewards for u_agents, test total rewards
        """
        u_agents_mean_rewards = [[] for _ in range(len(u_agents))]
        test_rewards = []

        for i in range(n_pairs):
            u_agent_idx = np.random.choice(len(u_agents))
            v_agent_idx = np.random.choice(len(v_agents))
            print(f'PAIR {i + 1} OF {n_pairs}')
            print('\n{:-^50}\n'.format(f'U_AGENT_{u_agent_idx} VS V_AGENT_{v_agent_idx}'))
            _, mean_rewards, _ = self._fit_agents(
                u_agents[u_agent_idx],
                v_agents[v_agent_idx],
                n_epochs=n_epochs,
                n_sessions=n_sessions,
                n_iter_update=n_iter_update,
                epsilon=-1,
                n_iter_debug=0,
            )
            print('\n{:-^50}\n'.format(''))

            u_agents_mean_rewards[u_agent_idx].append(np.min(mean_rewards))

        for i, u_agent in enumerate(u_agents):
            print(f'\nTESTING U_AGENT_{i}\n')
            test_rewards.append(
                self._test_agent(
                    u_agent=u_agent,
                    n_epochs=n_test_epochs,
                    n_sessions=n_sessions,
                    n_iter_update=n_iter_update,
                    epsilon=-1,
                ),
            )

        best_u_agent_idx = np.argmin([np.max(test) for test in test_rewards])
        print(f'\nBest agent is {best_u_agent_idx}, its test reward is {np.min(test_rewards[best_u_agent_idx])}\n')

        self.u_agent = u_agents[best_u_agent_idx]
        return u_agents[best_u_agent_idx], np.array(u_agents_mean_rewards), np.array(test_rewards)


def main():
    # env = gym.make('Pendulum-v0')
    # state_shape = env.observation_space.shape
    # action_shape = env.action_space.shape

    env = UnequalGame()
    u_agents = [CCEMAgent((2,), (1,), percentile_param=10, action_max=env.u_action_max, reward_param=-1) for _ in
                range(5)]
    v_agents = [CCEMAgent((2,), (1,), percentile_param=90, action_max=env.v_action_max, reward_param=1) for _ in
                range(5)]

    trainer = Trainer(u_agent=u_agents[0], env=env)

    trainer.fit_agents(
        v_agent=v_agents[0],
        n_epochs=100,
        n_sessions=100,
        n_iter_debug=50,
        n_iter_update=20,
        n_test_epochs=50,
    )

    trainer.fit_agents_one_by_one(
        v_agent=v_agents[0],
        n_epochs=100,
        n_sessions=100,
        n_iter_for_fit=50,
        n_iter_update=20,
        n_iter_debug=50,
        n_test_epochs=50,
    )

    trainer.fit_random_agent_pairs(
        u_agents,
        v_agents,
        n_pairs=2,
        n_epochs=10,
        n_test_epochs=50,
        n_sessions=100,
        n_iter_update=20,
    )

    trainer.test_agent(
        n_epochs=100,
        n_sessions=100,
        epsilon=-1,
        n_iter_update=20,
    )


if __name__ == '__main__':
    main()
