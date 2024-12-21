
from collections import defaultdict, deque
import torch
import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, capacity, observation_shape, action_shape):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape

        self.buffers = defaultdict(lambda: {
            'obs': deque(maxlen=capacity),
            'action': deque(maxlen=capacity),
            'reward': deque(maxlen=capacity),
            'next_obs': deque(maxlen=capacity),
            'done': deque(maxlen=capacity),
        })

    def push(self, agent_id, obs, action, reward, next_obs, done):
        self.buffers[agent_id]['obs'].append(obs)
        self.buffers[agent_id]['action'].append(action)
        self.buffers[agent_id]['reward'].append(reward)
        self.buffers[agent_id]['next_obs'].append(next_obs)
        self.buffers[agent_id]['done'].append(done)

    def sample(self, batch_size):
        all_agent_ids = list(self.buffers.keys())
        if not all_agent_ids:
            return None 

        total_transitions = sum(len(self.buffers[agent_id]['obs']) for agent_id in all_agent_ids)
        if total_transitions < batch_size:
            return None

        all_transitions = []
        for agent_id in all_agent_ids:
            agent_buffer = self.buffers[agent_id]
            for i in range(len(agent_buffer['obs'])):
                all_transitions.append({
                    'obs': agent_buffer['obs'][i],
                    'action': agent_buffer['action'][i],
                    'reward': agent_buffer['reward'][i],
                    'next_obs': agent_buffer['next_obs'][i],
                    'done': agent_buffer['done'][i]
                })

        indices = np.random.choice(len(all_transitions), batch_size, replace=False)

        obs_batch = np.array([all_transitions[i]['obs'] for i in indices])
        action_batch = np.array([all_transitions[i]['action'] for i in indices])
        reward_batch = np.array([all_transitions[i]['reward'] for i in indices])
        next_obs_batch = np.array([all_transitions[i]['next_obs'] for i in indices])
        done_batch = np.array([all_transitions[i]['done'] for i in indices])

        return {
            'obs': obs_batch,
            'action': action_batch,
            'reward': reward_batch,
            'next_obs': next_obs_batch,
            'done': done_batch
        }

    def update_last_reward(self, agent_id, new_reward):
        if agent_id not in self.buffers:
            return
        self.buffers[agent_id]['reward'][-1] = new_reward

    def __len__(self):
        return sum(len(self.buffers[agent_id]['obs']) for agent_id in self.buffers)

    def clear(self, agent_id=None):
        if agent_id:
            self.buffers[agent_id]['obs'].clear()
            self.buffers[agent_id]['action'].clear()
            self.buffers[agent_id]['reward'].clear()
            self.buffers[agent_id]['next_obs'].clear()
            self.buffers[agent_id]['done'].clear()
        else:
            for agent_id in self.buffers:
                self.clear(agent_id)