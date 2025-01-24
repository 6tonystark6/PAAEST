import torch


def whiten_and_color(content_feats, style_feats, beta=1.0):
    content = content_feats.squeeze(0)
    channels, height, width = content.shape
    content = content.reshape(channels, -1)

    content_mean = torch.mean(content, dim=1, keepdim=True)
    content -= content_mean
    content_cov = torch.mm(content, content.t()) / (height * width - 1)
    u_c, e_c, v_c = torch.svd(content_cov)

    rank_c = channels
    for i in range(channels):
        if e_c[i] < 0.00001:
            rank_c = i
            break
    content_sqrt_eigenvalues = e_c[:rank_c].pow(-0.5)

    whitening_matrix = torch.mm(v_c[:, :rank_c], torch.diag(content_sqrt_eigenvalues))
    whitening_matrix = torch.mm(whitening_matrix, v_c[:, :rank_c].t())
    whitened_content = torch.mm(whitening_matrix, content)

    style = style_feats.squeeze(0)
    channels, style_height, style_width = style.shape
    style = style.reshape(channels, -1)

    style_mean = torch.mean(style, dim=1, keepdim=True)
    style -= style_mean
    style_cov = torch.mm(style, style.t()) / (style_height * style_width - 1)
    u_s, e_s, v_s = torch.svd(style_cov)

    rank_s = channels
    for i in range(channels):
        if e_s[i] < 0.00001:
            rank_s = i
            break
    style_sqrt_eigenvalues = e_s[:rank_s].pow(0.5)

    coloring_matrix = torch.mm(v_s[:, :rank_s], torch.diag(style_sqrt_eigenvalues))
    coloring_matrix = torch.mm(coloring_matrix, v_s[:, :rank_s].t())
    colored_content = torch.mm(coloring_matrix, whitened_content) + style_mean

    final_colored_feats = colored_content.reshape(channels, height, width).unsqueeze(0).float()

    final_colored_feats = beta * final_colored_feats + (1.0 - beta) * content_feats
    return final_colored_feats
