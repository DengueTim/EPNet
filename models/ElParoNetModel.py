
import torch
import torch.nn as nn

class ElParoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.featureNet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 48, kernel_size=3),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.offsetNet = nn.Sequential(
            nn.Linear(1600,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256,256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Linear(128,4)
        )

    def forward(self, patch_a, patch_b):
        features_a = self.featureNet(patch_a)
        features_b = self.featureNet(patch_b)
        features = torch.cat((features_a, features_b),1)
        features = features.view(features.size(0), -1)
        return self.offsetNet(features)
