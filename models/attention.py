import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, query_input, key_value_input):
        query = self.query_conv(query_input)
        key = self.key_conv(key_value_input)
        value = self.value_conv(key_value_input)

        attention = torch.softmax(torch.matmul(query.flatten(2).transpose(1, 2), key.flatten(2)), dim=-1)
        attended_value = torch.matmul(attention, value.flatten(2).transpose(1, 2)).transpose(1, 2)
        attended_value = attended_value.view_as(value)

        fused_feature = self.mlp(attended_value)
        return fused_feature