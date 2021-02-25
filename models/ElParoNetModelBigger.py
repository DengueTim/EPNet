
import torch
import torch.nn as nn

class ElParoNetBigger(nn.Module):
    def __init__(self, patch_size):
        super().__init__()

        self.feature_net = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=7, stride=3),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 96, kernel_size=3),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 48, kernel_size=3),
            nn.BatchNorm2d(48),
            nn.LeakyReLU()
        )

        feature_size = int((patch_size - 6) / 3 - 2 - 2 - 2)
        feature_size = feature_size * feature_size * 48
        input_size = feature_size * 2

        self.offset_net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size // 2),
            nn.BatchNorm1d(input_size // 2),
            nn.LeakyReLU(),
            nn.Linear(input_size // 2, input_size // 2),
            nn.LeakyReLU(),
            nn.Linear(input_size // 2, input_size // 2),
            nn.LeakyReLU(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.LeakyReLU(),
            nn.Linear(input_size // 4, 6)
        )

    def forward(self, patch_a, patch_b):
        features_a = self.feature_net(patch_a)
        features_b = self.feature_net(patch_b)
        patches = torch.cat((features_a, features_b),1)
        patches = patches.view(patches.size(0),-1)
        return self.offset_net(patches)
