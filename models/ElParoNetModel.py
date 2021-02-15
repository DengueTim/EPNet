
import torch
import torch.nn as nn

class ElParoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.featureNet = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.LeakyReLU()
        )
        self.offsetNet = nn.Sequential(
            nn.Linear(2048 ,128),
            nn.LeakyReLU(),
            nn.Linear(128,128),
            nn.LeakyReLU(),
            nn.Linear(128,128),
            nn.LeakyReLU(),
            nn.Linear(128,4),
        )

    def forward(self, patch_a, patch_b):
        features_a = self.featureNet(patch_a)
        features_b = self.featureNet(patch_b)
        features = torch.cat((features_a, features_b),1)
        features = features.view(features.size(0), -1)
        return self.offsetNet(features)
