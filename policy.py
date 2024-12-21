import random
import torch

def epsilon_greedy_policy(observation, q_network, env, steps_done, EPS_START=1, EPS_END=0.1, EPS_DECAY=50, device="cpu"):
    """
    Implements an epsilon-greedy policy.

    Args:
        observation: The current observation of the environment.
        q_network: The Q-network used to estimate action values.
        env: The environment object.
        steps_done: The number of steps taken so far.
        EPS_START: The initial value of epsilon.
        EPS_END: The final value of epsilon.
        EPS_DECAY: The decay rate of epsilon.
        device: The device to run the policy on (e.g., "cpu" or "cuda").

    Returns:
        The action selected by the policy.
    """
    epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (steps_done / EPS_DECAY))
    
    if random.random() < epsilon:
        return env.action_space("red_0").sample()  # Explore
    else:
        observation = torch.Tensor(observation).to(device)
        with torch.no_grad():
            q_values = q_network(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]  # Exploit