import torch
import numpy as np
from copy import deepcopy


class Network(torch.nn.Module):
    """Neural network for an agent"""

    def __init__(self, input_shape, output_shape):
        """Create new network
        :param input_shape: input data shape
        :param output_shape: output data shape
        """
        super().__init__()
        self.linear_1 = torch.nn.Linear(input_shape[0], 50)
        self.linear_2 = torch.nn.Linear(50, 30)
        self.linear_3 = torch.nn.Linear(30, output_shape[0])
        self.relu = torch.nn.ReLU()
        self.tang = torch.nn.Tanh()

    def forward(self, input_):
        """Network step
        :param input_: input data
        :return: network output result
        """
        hidden = self.relu(self.linear_1(input_))
        hidden = self.relu(self.linear_2(hidden))
        output = self.tang(self.linear_3(hidden))
        return output


class CCEMAgent(torch.nn.Module):
    """Continuous cross-entropy method agent implementation"""

    def __init__(self, state_shape, action_shape, action_max, reward_param=1, percentile_param=70, noise_decrease=0.98,
                 tau=1e-2, learning_rate=1e-2, n_learning_per_fit=16, mini_batch_size=200):
        """Create new agent
        :param state_shape: environment's state shape
        :param action_shape: agent's action shape
        :param action_max: maximum action value
        :param reward_param: equal to 1 if agent wants to maximize reward otherwise -1
        :param percentile_param: percentile to get elite sessions
        :param noise_decrease: noise decrease value
        :param tau: network weights updating rate
        :param learning_rate: learning rate for gradient descent method
        :param n_learning_per_fit: number of network updating weights iterations per fit
        :param mini_batch_size: count of elements to sample to mini-batch
        """
        super().__init__()
        self.action_max = np.abs(action_max)
        self.reward_param = reward_param
        self.percentile_param = percentile_param
        self.noise_decrease = noise_decrease
        self.noise_threshold = 1
        self.min_noise_threshold = 0.1
        self.tau = tau
        self.n_learning_per_fit = n_learning_per_fit
        self.mini_batch_size = mini_batch_size
        self.network = Network(state_shape, action_shape)
        self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=learning_rate)

    def get_action(self, state, test=False):
        """Get an action by current state
        :param state: current environment state
        :param test: if True noise will not be added and will be otherwise
        :return: predicted action
        """
        state = torch.FloatTensor(state)
        predicted_action = self.network(state).detach().numpy() * self.action_max
        if not test:
            noise = self.noise_threshold * np.random.uniform(low=-self.action_max, high=self.action_max)
            predicted_action = np.clip(predicted_action + noise, -self.action_max, self.action_max)
        return predicted_action

    def get_elite_states_and_actions(self, sessions):
        """Select sessions with the most or least reward by percentile
        :param sessions: list of sessions to choose elite ones from
        :return: elite states, elite actions
        """
        total_rewards = [session['total_reward'] for session in sessions]
        reward_threshold = np.percentile(total_rewards, self.percentile_param)

        elite_states = []
        elite_actions = []
        for session in sessions:
            if self.reward_param * (session['total_reward'] - reward_threshold) > 0:
                elite_states.extend(session['states'])
                elite_actions.extend(session[f'{self}actions'])

        return torch.FloatTensor(elite_states), torch.FloatTensor(elite_actions)

    def learn_network(self, loss):
        """Update network weights by optimize loss
        :param loss: loss function to optimize
        :return: None
        """
        self.optimizer.zero_grad()
        old_network = deepcopy(self.network)
        loss.backward()
        self.optimizer.step()

        for new_parameter, old_parameter in zip(self.network.parameters(), old_network.parameters()):
            new_parameter.data.copy_(self.tau * new_parameter + (1 - self.tau) * old_parameter)

        return None

    def fit(self, sessions):
        """Fitting process using mini-batches
        :param sessions: sessions to fit on
        :return: None
        """
        elite_states, elite_actions = self.get_elite_states_and_actions(sessions)

        for _ in range(self.n_learning_per_fit):
            mini_batch_idxs = np.random.choice(range(elite_states.shape[0]), size=self.mini_batch_size)
            mini_batch_states = elite_states[mini_batch_idxs]
            mini_batch_actions = elite_actions[mini_batch_idxs]

            predicted_action = self.network(mini_batch_states) * self.action_max
            loss = torch.mean((predicted_action - mini_batch_actions) ** 2)
            self.learn_network(loss)

        if self.noise_threshold > self.min_noise_threshold:
            self.noise_threshold *= self.noise_decrease

        return None

    def __str__(self):
        """An agent string representation to define if it is u_agent or v_agent
        :return: string representation
        """
        return 'u_' if self.reward_param == -1 else 'v_'
