import torch.nn as nn


class MLPReg(nn.Module):
    def __init__(self, input, dim=512):
        super(MLPReg, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(len(input[0][0]), dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
        )

    def forward(self, x):
        return self.layer(x)
