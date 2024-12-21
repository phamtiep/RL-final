
from optimize import optimize_model
from policy import epsilon_greedy_policy as policy
import torch

def save_model(i_episode, policy_net, target_net, optimizer, episode_rewards, episode_losses, path):
    torch.save({
        'episode': i_episode,
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
    }, path)

def train_agents(env, num_episodes, policy_net, target_net, red_policy_net, buffer, optimizer, steps_done, episode, pretrained_net, TAU, BATCH_SIZE, GAMMA,
 EPS_START=1, EPS_END=0.1, EPS_DECAY=50, device="cpu"):
    episode_rewards = []
    episode_losses = []
    env.reset()
    for i_episode in range(episode, num_episodes):
        env.reset()
        episode_reward = 0
        running_loss = 0.0
        steps_done += 1
        

        for agent in env.agent_iter():

            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation
            episode_reward += reward

            if done:
                action = None
                env.step(action)
            else:
                agent_handle = agent.split("_")
                agent_id = agent_handle[1]
                agent_team = agent_handle[0]
                if agent_team == "blue":

                    buffer.update_last_reward(agent_id, reward)

                    action = policy( observation,policy_net, env, steps_done,  EPS_START=EPS_START, EPS_END=EPS_END, EPS_DECAY=EPS_DECAY, device=device )
                    env.step(action)

                    try:
                        next_observation = env.observe(agent)
                        agent_done = False
                    except:
                        next_observation = None
                        agent_done = True

                    reward = 0

                    buffer.push(agent_id, observation, action, reward, next_observation, agent_done)

                    optimize_model(policy_net, target_net, optimizer, buffer, BATCH_SIZE, GAMMA, device)

                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                    target_net.load_state_dict(target_net_state_dict)

                else:
                    action = policy( observation,red_policy_net, env, steps_done,  EPS_START=EPS_START, EPS_END=EPS_END, EPS_DECAY=EPS_DECAY, device=device )
                    env.step(action)
            if i_episode % 4 == 0 and i_episode < 24:
                red_policy_net.load_state_dict(policy_net.state_dict())
            elif i_episode == 24:
                red_policy_net.load_state_dict(pretrained_net.state_dict())
        episode_rewards.append(episode_reward)
        episode_losses.append(running_loss)

        print(f'Episode {i_episode + 1}/{num_episodes}')
        print(f'Total Reward of previous episode: {episode_reward:.2f}')
        print(f'Average Loss: {running_loss:.4f}')
        print(f'Epsilon: {linear_epsilon(steps_done)}')
        print('-' * 40)
        save_model(i_episode, policy_net, target_net, optimizer, episode_rewards, episode_losses, path=f"models/blue_{i_episode}.pt")
                