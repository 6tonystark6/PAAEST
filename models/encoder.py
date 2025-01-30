from torch import nn
from torchvision.models import vgg19
from attention import AttentionFusion


class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = vgg[:2]
        self.slice2 = vgg[2:7]
        self.slice3 = vgg[7:12]
        self.slice4 = vgg[12:21]
        for p in self.parameters():
            p.requires_grad = False

        self.avgpool_h2_h3 = nn.AdaptiveAvgPool2d((64, 64))
        self.fc_h2_h3 = nn.Conv2d(128, 256, kernel_size=1)

        self.avgpool_fused_h4 = nn.AdaptiveAvgPool2d((32, 32))
        self.fc_fused_h4 = nn.Conv2d(256, 512, kernel_size=1)

        self.attention_h2_h3 = AttentionFusion(256, 256)
        self.attention_fused_h4 = AttentionFusion(512, 512)

    def forward(self, images, output_last_feature=False):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)

        h2_resized = self.fc_h2_h3(self.avgpool_h2_h3(h2))
        fused_h2_h3 = self.attention_h2_h3(h2_resized, h3)

        fused_resized = self.fc_fused_h4(self.avgpool_fused_h4(fused_h2_h3))
        fused_feature = self.attention_fused_h4(fused_resized, h4)

        if output_last_feature:
            return fused_feature
        else:
            return h1, h2, fused_h2_h3, fused_feature
