import torch
from utils import get_action
from utils import convert_obs
def gather_training_data(env, episode, base_q_network, better_agent):
    env.reset()
    X, y = [], []

    # Buffer for storing agent data
    buffer = {}

    # Gathering training data
    for id, agent in enumerate(env.agent_iter()):
        observation, reward, termination, truncation, info = env.last()

        agent_handle = agent.split("_")[0]
        agent_id = int(agent.split("_")[1])
        if termination or truncation:
            action = None  # This agent has died
        else:
            if agent_handle == "red":
                action = get_action(env, episode, agent, observation, base_q_network, 'best')
            else:
                action = get_action(env, episode, agent, observation, better_agent, 'epsilon')

        if agent not in buffer:
            buffer[agent] = []
        buffer[agent].append((agent, observation, action, reward, termination, truncation, info))
        env.step(action)

    # Process buffer and prepare training data
    for agent in buffer.keys():
        state_array = buffer[agent]
        agent_handle = agent.split("_")[0]

        for i in range(0, len(state_array)):
            agent, observation, action, prv_reward, termination, truncation, info = state_array[i]
            _1, nxt_observation, nxt_action, reward, nxt_termination, nxt_truncation, nxt_info = state_array[i]
            if i < len(state_array) - 1:
                _1, nxt_observation, nxt_action, reward, nxt_termination, nxt_truncation, nxt_info = state_array[i + 1]

            observation = convert_obs(observation)
            nxt_observation = convert_obs(nxt_observation)

            with torch.no_grad():
                next_max = better_agent(nxt_observation).squeeze(dim=0).max()
                tmp = better_agent(observation).squeeze(dim=0)
                if i == len(state_array):
                    next_max = 0
                tmp[action] = reward + next_max
                X.append(observation.squeeze(dim=0))
                y.append(tmp)

    return X, y
