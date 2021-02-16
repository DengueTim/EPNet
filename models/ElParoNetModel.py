
import torch
import torch.nn as nn

class ElParoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(578,512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Linear(128,4)
        )

    def forward(self, patch_a, patch_b):
        patches = torch.cat((patch_a, patch_b),1)
        patches = patches.view(patches.size(0),-1)
        return self.net(patches)
