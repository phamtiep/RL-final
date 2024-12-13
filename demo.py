from magent2.environments import battle_v4
import os
import cv2
import sys
import torch

if __name__ == "__main__":
    # Kiểm tra xem có CUDA hay không, nếu không thì sử dụng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo môi trường và thiết lập
    demo_env = battle_v4.env(map_size=45, max_cycles=300, render_mode="rgb_array")
    demo_env.reset()
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 32
    frames = []
    
    # Pretrained policies
    import base_model
    import resnet
    
    q_network = resnet.QNetwork(
        demo_env.observation_space("blue_0").shape, demo_env.action_space("blue_0").n
    )
    
    # Chuyển mô hình lên GPU nếu có
    q_network.to(device)
    
    q_network.load_state_dict(
        torch.load("model/agent_kamikaze.pth", weights_only=True, map_location=device)
    )



    red_cnt = 81
    blue_cnt = 81
    mem = {}
    step = 0
    for agent in demo_env.agent_iter():

        observation, reward, termination, truncation, info = demo_env.last()
        
        agent_handle = agent.split("_")[0]
        if termination or truncation:
            
            if (termination and agent not in mem):
                mem[agent] = 1
                if (agent_handle == 'red'): red_cnt -= 1
                else: blue_cnt -= 1

            action = None  # this agent has died
        else:
            if agent_handle == "blue":
                # Chuyển observation lên GPU nếu cần
                observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0).to(device)
                )
                with torch.no_grad():
                    q_values = q_network(observation)
                action = torch.argmax(q_values, dim=1).cpu().numpy()[0]  # Chuyển kết quả trở lại CPU
            else:
                action = demo_env.action_space(agent).sample()
        
        demo_env.step(action)
        if (step == 162):
            frames.append(demo_env.render())
            step = 0
        step += 1

    print(red_cnt, blue_cnt)

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"battle_best.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print("Done recording battle !!!")

    demo_env.close()
