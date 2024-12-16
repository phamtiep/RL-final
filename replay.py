import torch
from utils import get_action
from utils import convert_obs

def gather_training_data(env, episode, base_q_network, better_agent):
    # Kiểm tra và chọn thiết bị (GPU hoặc CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Di chuyển mô hình lên GPU (nếu có)
    base_q_network.to(device)
    better_agent.to(device)

    env.reset()
    X, y = [], []

    # Buffer để lưu trữ dữ liệu của các agent
    buffer = {}

    # Thu thập dữ liệu huấn luyện
    for id, agent in enumerate(env.agent_iter()):
        observation, reward, termination, truncation, info = env.last()

        agent_handle = agent.split("_")[0]
        agent_id = int(agent.split("_")[1])
        if termination or truncation:
            action = None  # Agent này đã chết
        else:
            if agent_handle == "red":
                action = get_action(env, episode, agent, observation, base_q_network, 'best')
            else:
                action = get_action(env, episode, agent, observation, better_agent, 'epsilon')

        if agent not in buffer:
            buffer[agent] = []
        buffer[agent].append((agent, observation, action, reward, termination, truncation, info))
        env.step(action)

    # Xử lý buffer và chuẩn bị dữ liệu huấn luyện
    for agent in buffer.keys():
        state_array = buffer[agent]
        agent_handle = agent.split("_")[0]

        for i in range(0, len(state_array)):
            agent, observation, action, prv_reward, termination, truncation, info = state_array[i]
            _1, nxt_observation, nxt_action, reward, nxt_termination, nxt_truncation, nxt_info = state_array[i]
            if i < len(state_array) - 1:
                _1, nxt_observation, nxt_action, reward, nxt_termination, nxt_truncation, nxt_info = state_array[i + 1]

            # Chuyển observation và next observation lên GPU
            observation = convert_obs(observation).to(device)
            nxt_observation = convert_obs(nxt_observation).to(device)

            # Tính toán với mô hình
            with torch.no_grad():
                next_max = better_agent(nxt_observation).squeeze(dim=0).max()
                tmp = better_agent(observation).squeeze(dim=0)
                if i == len(state_array):
                    next_max = 0
                tmp[action] = reward + next_max

                # Chuyển tensor X và y lên GPU nếu cần
                X.append(observation.squeeze(dim=0))
                y.append(tmp)

    return X, y
