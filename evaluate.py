import torch_fidelity
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image

folder1 = './output/transfer1'
folder2 = './output/transfer2'
folder3 = './test_content_resized'

metrics = torch_fidelity.calculate_metrics(
    input1=folder1,
    input2=folder3,
    cuda=True,
    isc=False,
    fid=True,
    kid=True,
    verbose=False
)

psnr_values = []
ssim_values = []

for i in range(1, 1001):
    filename = f"{i}.jpg"
    path2 = os.path.join(folder2, filename)
    path3 = os.path.join(folder3, filename)

    if not os.path.exists(path2) or not os.path.exists(path3):
        print(f"file {filename} doesn't exist, skip...")
        continue

    img2 = np.array(Image.open(path2).convert("RGB"))
    img3 = Image.open(path3).convert("RGB")

    width, height = img3.size
    left = (width - 256) // 2
    top = (height - 256) // 2
    right = left + 256
    bottom = top + 256
    img3_cropped = img3.crop((left, top, right, bottom))
    img3_cropped = np.array(img3_cropped)

    if img2.shape != img3_cropped.shape:
        print(f"file {filename} size doesn't match, skip...")
        continue

    psnr_value = psnr(img2, img3_cropped, data_range=img2.max() - img2.min())
    ssim_value = ssim(img2, img3_cropped, data_range=img2.max() - img2.min(), multichannel=True)

    psnr_values.append(psnr_value)
    ssim_values.append(ssim_value)

    print(f"figure {filename} PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}")

psnr_mean = np.mean(psnr_values)
ssim_mean = np.mean(ssim_values)
print("Metrics:")
print(f"FID: {metrics['frechet_inception_distance']}")
print(f"KID: {metrics['kernel_inception_distance_mean']}")
print(f"PSNR: {psnr_mean:.4f}")
print(f"SSIM: {ssim_mean:.4f}")