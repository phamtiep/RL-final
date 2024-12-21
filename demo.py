from magent2.environments import battle_v4
import os
import cv2


if __name__ == "__main__":
    env = battle_v4.env(map_size=45, render_mode="rgb_array", max_cycles = 1000)
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 35
    frames = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    myAgent = QNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n).to(device) # Move model to device
    myAgent.load_state_dict(
        torch.load("models/blue_dqn_final.pt", map_location=device, weights_only=True) # Load weights to the same device
    )
    # random policies
    env.reset()
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # this agent has died
        else:
            agent_handle = agent.split("_")[0]
            observation = (
            torch.Tensor(obs).float().to(device)
            )
            if agent_handle == "red":
                with torch.no_grad():
                   action = env.action_space(agent).sample()
            else:
                with torch.no_grad():
                  
                  q_values_a = myAgent(observation)
                  action = torch.argmax(q_values_a, dim=1).cpu().numpy()[0]


        env.step(action)

        if agent == "red_0":
            frames.append(env.render())

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"random.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print("Done recording random agents")

    # pretrained policies
    frames = []
    env.reset()


    q_network = BaseNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load("models/red.pt", weights_only=True, map_location="cpu")
    )


    for agent in env.agent_iter():

        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # this agent has died
        else:
            agent_handle = agent.split("_")[0]
            
            if agent_handle == "red":
                with torch.no_grad():
                  observation = (
                    torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0)
                    )
                  q_values= q_network(observation)
                  action = torch.argmax(q_values, dim=1).cpu().numpy()[0]
            else:
                with torch.no_grad():
                  observation = (
                  torch.Tensor(obs).float().to(device)
                  )
                  q_values_a= myAgent(observation)
                  action = torch.argmax(q_values_a, dim=1).cpu().numpy()[0]

        env.step(action)

        if agent == "red_0":
            frames.append(env.render())

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"pretrained.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print("Done recording pretrained agents")

    env.close()