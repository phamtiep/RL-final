import torch
import torch.nn as nn
import numpy as np
def optimize_model(policy_net, target_net, optimizer, buffer, BATCH_SIZE, GAMMA, device):

    batch = buffer.sample(BATCH_SIZE)

    if batch is None:
        return 0 

    state_batch = torch.from_numpy(batch['obs']).float().to(device)
    action_batch = torch.from_numpy(batch['action']).long().to(device)
    reward_batch = torch.from_numpy(batch['reward']).float().to(device)
    next_state_batch = torch.from_numpy(batch['next_obs']).float().to(device)
    done_batch = torch.from_numpy(batch['done']).float().to(device)

    action_batch = action_batch.unsqueeze(1)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    non_final_mask = (done_batch == 0).squeeze()

    if non_final_mask.any():
        next_state_values[non_final_mask] = target_net(next_state_batch[non_final_mask]).max(1).values.detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    
    optimizer.step()
    
    return loss.item()