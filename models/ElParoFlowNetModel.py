
import torch
import torch.nn as nn
import torch.nn.functional as F

class ElParoFlowNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7),
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

        self.offset_net = nn.Sequential(
            nn.Conv2d(64, 1024, kernel_size=7),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 2, kernel_size=1),
        )

    def forward(self, patch_a, patch_b):
        features_a = self.feature_net(patch_a)
        features_b = self.feature_net(patch_b)
        # Concat along channel dim..
        features_ab = torch.cat((features_a, features_b), dim=1)
        return self.offset_net(features_ab)
