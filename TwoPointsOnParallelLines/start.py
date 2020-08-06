from TwoPointsOnParallelLines.two_points_on_parallel_lines_env import TwoPointsOnParallelLines
import matplotlib.pyplot as plt
from TwoPointsOnParallelLines.optimal_agents import OptimalUAgent, OptimalVAgent
from TwoPointsOnParallelLines.other_agents import SinCosUAgent, NormUAgent, OptimalConstantCounterVAgent, ConstantVAgent, SinVAgent


env = TwoPointsOnParallelLines()

u_agent = NormUAgent(env)
v_agent = OptimalConstantCounterVAgent(env, u_agent)

states = []
u_actions = []
v_actions = []

total_reward = 0
state = env.reset()
done = False

while not done:
    states.append(state)
    u_action = u_agent.get_action(state)
    u_actions.append(u_action)
    v_action = v_agent.get_action(state)
    v_actions.append(v_action)
    state, reward, done, _ = env.step(u_action, v_action)
    total_reward += reward

print('total_reward={}'.format(total_reward))

plt.plot([state[0] for state in states], [state[1] for state in states])
plt.plot([state[0] for state in states], [state[3] for state in states])
plt.show()

plt.plot([state[0] for state in states], u_actions)
plt.plot([state[0] for state in states], v_actions)
plt.show()
