import torch


def whiten_and_color(content_feature, style_feature, beta=1.0):
    batch_size, c, ch, cw = content_feature.shape
    content_feature = content_feature.view(batch_size, c, -1)
    content_mean = torch.mean(content_feature, 2, keepdim=True)
    content_feature = content_feature - content_mean

    content_cov = torch.bmm(content_feature, content_feature.transpose(1, 2)) / (ch * cw - 1)
    content_u, content_e, content_v = torch.svd(content_cov)

    k_c = c
    for i in range(c):
        if content_e[:, i].min() < 0.00001:
            k_c = i
            break
    content_d = content_e[:, :k_c].pow(-0.5)

    w_step1 = torch.bmm(content_v[:, :, :k_c], torch.diag_embed(content_d))
    w_step2 = torch.bmm(w_step1, content_v[:, :, :k_c].transpose(1, 2))
    whitened = torch.bmm(w_step2, content_feature)

    style_feature = style_feature.view(batch_size, c, -1)
    style_mean = torch.mean(style_feature, 2, keepdim=True)
    style_feature = style_feature - style_mean

    style_cov = torch.bmm(style_feature, style_feature.transpose(1, 2)) / (ch * cw - 1)
    style_u, style_e, style_v = torch.svd(style_cov)

    k_s = c
    for i in range(c):
        if style_e[:, i].min() < 0.00001:
            k_s = i
            break
    style_d = style_e[:, :k_s].pow(0.5)

    c_step1 = torch.bmm(style_v[:, :, :k_s], torch.diag_embed(style_d))
    c_step2 = torch.bmm(c_step1, style_v[:, :, :k_s].transpose(1, 2))
    colored = torch.bmm(c_step2, whitened) + style_mean

    colored_feature = colored.view(batch_size, c, ch, cw).float()

    colored_feature = beta * colored_feature + (1.0 - beta) * content_feature.view(batch_size, c, ch, cw)
    return colored_feature
