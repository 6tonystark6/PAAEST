import torch
import torch.nn.functional as F
from torch.autograd import Variable


def gaussian_kernel(size, sigma):
    axis = torch.arange(0, size).float()
    mean = size // 2
    kernel = torch.exp(-0.5 * (axis - mean) ** 2 / sigma ** 2)
    return kernel / kernel.sum()


def generate_window(window_size, channels):
    kernel_1d = gaussian_kernel(window_size, 1.5).unsqueeze(1)
    kernel_2d = kernel_1d @ kernel_1d.t()
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    window = Variable(kernel_2d.expand(channels, 1, window_size, window_size).contiguous())
    return window


def compute_ssim(img1, img2, window, window_size, channels, average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    # mu1_mu2 = mu1_sq * mu2_sq
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = 2 * (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if average else ssim_map.view(ssim_map.size(1), -1).mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, average=True):
    img1, img2 = torch.clamp(img1, 0, 1), torch.clamp(img2, 0, 1)
    _, channels, _, _ = img1.size()
    window = generate_window(window_size, channels)
    if img1.is_cuda:
        window = window.cuda(img1.device)
    window = window.type_as(img1)
    return compute_ssim(img1, img2, window, window_size, channels, average)


def calc_mean_std(features):
    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def calc_covariance_matrix(features):
    b, c, h, w = features.size()
    features_flat = features.view(b, c, -1)
    covariance = torch.bmm(features_flat, features_flat.transpose(1, 2)) / (h * w)
    return covariance
