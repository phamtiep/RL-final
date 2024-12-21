import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
        )
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
        )

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        if len(x.shape) == 3:
            batchsize = 1
            x = x.unsqueeze(0)
        else:
            batchsize = x.shape[0]
        x = torch.fliplr(x).permute(0,3,1,2)
        x = self.cnn(x)
        x = x.reshape(batchsize, -1)
        return self.network(x)

test = QNetwork((13,13,5), 21)
test_obs = torch.rand((13,13,5))
test(test_obs)