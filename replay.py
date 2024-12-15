import torch
from utils import get_action
from utils import convert_obs
def replay(episode, env, base_q_network, agent):
  best_score = 100
  episodes = 300
  for episode in range (1, episodes + 1):

      print(f'Episode number {episode} running...................................')
      
      #gathering training data
      env.reset()
      X, y = [], []

      #because the reward is last reward, not te current reward so we have to trace backward
      buffer: dict[str, list[tuple]] = {}
      for id, agent in enumerate(env.agent_iter()):
          observation, reward, termination, truncation, info = env.last()
          
          agent_handle = agent.split("_")[0]
          agent_id = int(agent.split("_")[1])
          if termination or truncation:
              action = None  # this agent has died
          else:
              if agent_handle == "red":
                  action = get_action(env, episode, agent, observation, base_q_network, 'random')
              else:
                  action = get_action(env, episode, agent, observation, agent, 'epsilon')
          
          if (agent not in buffer):
              buffer[agent] = []
          buffer[agent].append((agent, observation, action, reward, termination, truncation, info))
          env.step(action)

      for agent in buffer.keys():
          state_array = buffer[agent]
          agent_handle = agent.split("_")[0]
          #if (agent_handle == 'red'): continue

          for i in range(0, len(state_array)):
              agent, observation, action, prv_reward, termination, truncation, info = state_array[i]
              _1, nxt_observation, nxt_action, reward, nxt_termination, nxt_truncation, nxt_info = state_array[i]
              if (i < len(state_array) - 1):
                  _1, nxt_observation, nxt_action, reward, nxt_termination, nxt_truncation, nxt_info = state_array[i + 1]

              observation = convert_obs(observation)
              nxt_observation = convert_obs(nxt_observation)

              with torch.no_grad():
                  next_max = agent(nxt_observation).squeeze(dim = 0).max()
                  tmp = agent(observation).squeeze(dim = 0)
                  if (i == len(state_array)):
                      next_max = 0
                  tmp[action] = reward + next_max
                  X.append(observation.squeeze(dim = 0))
                  y.append(tmp)
      return [X,y]