import random
import torch

def epsilon_greedy_policy(observation, q_network, env, steps_done, EPS_START=1, EPS_END=0.1, EPS_DECAY=50, device="cpu"):

    epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (steps_done / EPS_DECAY))
    
    if random.random() < epsilon:
        return env.action_space("red_0").sample()
    else:
        observation = torch.Tensor(observation).to(device)
        with torch.no_grad():
            q_values = q_network(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]