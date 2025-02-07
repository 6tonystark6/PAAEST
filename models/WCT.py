import torch


def whiten_and_color(content_feature, style_feature, eps=1e-3):
    B, C, H, W = content_feature.shape
    content_feature = content_feature.view(B, C, -1)  # [B, C, N]
    style_feature = style_feature.view(B, C, -1)

    content_mean = content_feature.mean(dim=2, keepdim=True)
    style_mean = style_feature.mean(dim=2, keepdim=True)

    content_centered = content_feature - content_mean
    style_centered = style_feature - style_mean

    content_cov = (content_centered @ content_centered.transpose(1, 2)) / (H * W - 1) + torch.eye(C, device=content_feature.device) * eps
    style_cov = (style_centered @ style_centered.transpose(1, 2)) / (H * W - 1) + torch.eye(C, device=style_feature.device) * eps

    L_c = torch.linalg.cholesky(content_cov)
    L_c_inv = torch.linalg.inv(L_c)
    whitened = L_c_inv @ content_centered
    L_s = torch.linalg.cholesky(style_cov)
    colored = L_s @ whitened

    colored_feature = colored + style_mean

    return colored_feature.view(B, C, H, W)
