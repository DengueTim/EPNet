
import torch
import torch.nn as nn


class ElParoFlowNetCvEa(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_flow = 8
        self.flow_size = self.max_flow * 2 + 1
        self.cv_size = self.flow_size ** 2

        self.feature_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 24, kernel_size=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24, 24, kernel_size=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.Conv2d(24, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )

        self.flow_net = nn.Sequential(
            nn.Conv3d(self.flow_size, self.cv_size, (self.flow_size, 1, 1)),
            nn.BatchNorm3d(self.cv_size),
            nn.LeakyReLU(),
            nn.Conv3d(self.cv_size, self.cv_size // 2, 1),
            nn.BatchNorm3d(self.cv_size // 2),
            nn.LeakyReLU(),
            nn.Conv3d(self.cv_size // 2, self.cv_size // 4, 1),
            nn.BatchNorm3d(self.cv_size // 4),
            nn.LeakyReLU(),
            nn.Conv3d(self.cv_size // 4, self.cv_size // 2, (1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(self.cv_size // 2),
            nn.LeakyReLU(),
            nn.Conv3d(self.cv_size // 2, self.cv_size // 4, 1),
            nn.BatchNorm3d(self.cv_size // 4),
            nn.LeakyReLU(),
            nn.Conv3d(self.cv_size // 4, self.cv_size // 8, 1),
            nn.BatchNorm3d(self.cv_size // 8),
            nn.LeakyReLU(),
            nn.Conv3d(self.cv_size // 8, self.cv_size // 2, (1, 3, 3)),
            nn.BatchNorm3d(self.cv_size // 2),
            nn.LeakyReLU(),
            nn.Conv3d(self.cv_size // 2, self.cv_size // 4, 1),
            nn.BatchNorm3d(self.cv_size // 4),
            nn.LeakyReLU(),
            nn.Conv3d(self.cv_size // 4, self.cv_size // 8, 1),
            nn.BatchNorm3d(self.cv_size // 8),
            nn.LeakyReLU(),
            nn.Conv3d(self.cv_size // 8, self.cv_size // 16, 1),
            nn.BatchNorm3d(self.cv_size // 16),
            nn.LeakyReLU(),
            nn.Conv3d(self.cv_size // 16, 2, 1),
        )

    def forward(self, patch_a, patch_b):
        features_a = self.feature_net(patch_a)
        features_b = self.feature_net(patch_b)

        (bs, feats, h, w) = features_a.size()
        mv = torch.zeros((bs, self.flow_size, self.flow_size, h, w)).to(features_a.device)

        # Compute 'match' volume.
        for yi in range(-self.max_flow, self.max_flow + 1):
            yix = max(0,yi)
            yin = min(yi,0)
            for xi in range(-self.max_flow, self.max_flow + 1):
                xix = max(0,xi) # 0 to 8
                xin = min(xi,0) # -8 to 0

                features_a_cropped = features_a[:, :,  yix:h + yin, xix:w + xin]
                features_b_cropped = features_b[:, :, -yin:h - yix, -xin:w - xix]
                features_difference = features_a_cropped - features_b_cropped
                mv[:, yi + self.max_flow, xi + self.max_flow, yix:h + yin, xix:w + xin] = \
                    torch.reciprocal(torch.sum(torch.abs(features_difference), dim=1))

        return self.flow_net(mv).squeeze()
