


def train_agents(env, num_episodes, policy_net, target_net, red_policy_net, buffer, optimizer, steps_done, episode, pretrained_net):
    episode_rewards = []
    episode_losses = []
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
                action = None  # Agent is dead
                env.step(action)
            else:
                agent_handle = agent.split("_")
                agent_id = agent_handle[1]
                agent_team = agent_handle[0]
                if agent_team == "blue":

                    buffer.update_last_reward(agent_id, reward) # update reward of last agent's action (bad environment!)

                    action = policy(observation, policy_net)
                    env.step(action)

                    try:
                        next_observation = env.observe(agent)
                        agent_done = False
                    except:
                        next_observation = None
                        agent_done = True

                    reward = 0 # Wait for next time to be selected to get reward

                    # Store the transition in buffer
                    buffer.push(agent_id, observation, action, reward, next_observation, agent_done)

                    # Perform one step of the optimization (on the policy network)
                    loss = optimize_model(buffer, policy_net, target_net, optimizer)
                    if loss is not None:
                        running_loss += loss

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_