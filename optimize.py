import torch
import torch.nn as nn
import numpy as np
def optimize_model(policy_net, target_net, optimizer, buffer, BATCH_SIZE, GAMMA, device):
    """
    Optimizes the policy network using a batch of experiences from the replay buffer.
    
    Args:
        policy_net: The policy network to optimize.
        target_net: The target network used for calculating target Q-values.
        optimizer: The optimizer for the policy network.
        buffer: The replay buffer containing experiences.
        BATCH_SIZE: The batch size for training.
        GAMMA: The discount factor.
        device: The device to use for computations (CPU or GPU).
    
    Returns:
        The loss value for the current batch.
    """
    
    # Sample a batch from the buffer
    batch = buffer.sample(BATCH_SIZE)
    
    # Handle cases where the buffer doesn't have enough samples yet
    if batch is None:
        return 0  # Return 0 loss if no batch is sampled
    
    # Unpack the batch
    state_batch = torch.from_numpy(batch['obs']).float().to(device)
    action_batch = torch.from_numpy(batch['action']).long().to(device)
    reward_batch = torch.from_numpy(batch['reward']).float().to(device)
    next_state_batch = torch.from_numpy(batch['next_obs']).float().to(device)
    done_batch = torch.from_numpy(batch['done']).float().to(device)
    
    # Reshape action_batch to (BATCH_SIZE, 1) for gather()
    action_batch = action_batch.unsqueeze(1)
    
    # Calculate Q-values for the current state-action pairs
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Calculate target Q-values
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    non_final_mask = (done_batch == 0).squeeze()  # Create a mask for non-terminal states
    
    # Only compute for non-terminal states
    if non_final_mask.any():
        next_state_values[non_final_mask] = target_net(next_state_batch[non_final_mask]).max(1).values.detach()
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    
    optimizer.step()
    
    return loss.item()  # Return the loss value