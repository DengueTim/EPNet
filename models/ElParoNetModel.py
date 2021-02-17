
import torch
import torch.nn as nn

class ElParoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2178,1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128,4)
        )

    def forward(self, patch_a, patch_b):
        patches = torch.cat((patch_a, patch_b),1)
        patches = patches.view(patches.size(0),-1)
        return self.net(patches)
