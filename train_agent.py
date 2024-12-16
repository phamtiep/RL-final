from magent2.environments import battle_v4
import torch.optim as optim
from torch import nn
import os
import numpy as np
import random
import sys
from torch.utils.data import DataLoader, TensorDataset
import time
from replay import gather_training_data
from utils import *
from evaluate_fight import evaluate
from train_model import train_model

def debug(var):
    print(var)
    sys.exit()

start_time = time.time()

# Cấu hình môi trường

env = battle_v4.env(map_size=45, minimap_mode=False, step_reward=-0.005,
dead_penalty=-1, attack_penalty=-0.1, attack_opponent_reward=1,
max_cycles=300, extra_features=False, render_mode = "rgb_array")
num_agent = 162
env.reset()
vid_dir = "video"
os.makedirs(vid_dir, exist_ok=True)
fps = 35
frames = []

#=========================== Hyperparams setting =======================================================================
# pretrained policies
frames = []
env.reset()
import base_model
import resnet
import torch

# training hyper params
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Base model
base_q_network = base_model.QNetwork(
    env.observation_space("red_0").shape, env.action_space("red_0").n
).to(device)

base_q_network.load_state_dict(
    torch.load("model/red.pt", weights_only=True, map_location=device)
)

# Current training model
better_agent = resnet.QNetwork(
    env.observation_space("blue_0").shape, env.action_space("blue_0").n
).to(device)
lr = 0.001
optimizer = optim.Adam(better_agent.parameters(), lr=lr)
loss_function = nn.MSELoss()

def run_episodes(env, episodes, base_q_network, better_agent, optimizer, lr, loss_function, best_score=100):
    for episode in range(1, episodes + 1):
        print(f'Episode number {episode} running...................................')

        # Thu thập dữ liệu huấn luyện
        X, y = gather_training_data(env, episode, base_q_network, better_agent)

        # Chuyển đổi dữ liệu huấn luyện thành tensor và chuyển lên GPU
        X_tensor = torch.stack(X).to(device)
        y_tensor = torch.stack(y).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

        # Huấn luyện mô hình
        train_model(100, dataloader, better_agent, optimizer, lr, loss_function)

        # Đánh giá và lưu mô hình nếu nó có sự cải thiện
        if episode % 3 == 0:
            avg = evaluate(red_agent=base_q_network, blue_agent=better_agent, rounds=5, debug=True)
            if avg < best_score:
                torch.save(better_agent.state_dict(), 'model/myagent.pth')
                print('Agent saved !')
                print()
                best_score = avg
    return best_score


best_score = run_episodes(
    env, episodes=50, 
    base_q_network=base_q_network, 
    better_agent=better_agent,
    optimizer=optimizer, 
    lr=lr, 
    loss_function=loss_function,
)

print(best_score)
env.close()
#===================================================================================================================

end_time = time.time()

print(f'Total running time : {(end_time - start_time)/3600}hrs')
