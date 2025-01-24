import torch
import torch.nn as nn
import torch.nn.functional as F


class RC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activated = activated

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            return F.relu(h)
        else:
            return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = RC(512, 256, 3, 1)
        self.rc2 = RC(256, 256, 3, 1)
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 256, 3, 1)
        self.rc5 = RC(256, 128, 3, 1)
        self.rc6 = RC(128, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.rc8 = RC(64, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, 1, False)

        self.interpolation_layer_1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.mlp_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        )
        self.interpolation_layer_2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.mlp_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, features, mixed_feature_1, mixed_feature_2):
        h = self.rc1(features)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc2(h)
        h = self.rc3(h)
        h = self.rc4(h)

        combined_features_1 = torch.cat([h, mixed_feature_2], dim=1)
        W1 = self.interpolation_layer_1(combined_features_1)
        fused_feature_1 = mixed_feature_2 * W1 + h * (1 - W1)
        h = self.mlp_1(fused_feature_1)

        h = self.rc5(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc6(h)

        combined_features_2 = torch.cat([h, mixed_feature_1], dim=1)
        W2 = self.interpolation_layer_2(combined_features_2)
        fused_feature_2 = mixed_feature_1 * W2 + h * (1 - W2)
        h = self.mlp_2(fused_feature_2)

        h = self.rc7(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc8(h)
        h = self.rc9(h)
        return h
