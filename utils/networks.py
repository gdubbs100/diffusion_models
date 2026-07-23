import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        ## input dim is dims of points + 1 for time idx
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, t):
        t_expanded = t.expand(x.size(0),  x.size(1), 1)
        x = torch.cat([x, t_expanded], dim = -1)
        return self.net(x)