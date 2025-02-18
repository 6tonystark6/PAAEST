import torch_fidelity
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os

input_dir = './test_content_resized'
output_dir = './test_content_resized1'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    input_file_path = os.path.join(input_dir, filename)

    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        with Image.open(input_file_path) as img:
            img_resized = img.resize((512, 512))

            output_file_path = os.path.join(output_dir, filename)

            img_resized.save(output_file_path)
            print(f"image {filename} resized and saved to {output_file_path}")

folder1 = './output/transfer1'
folder2 = './output/transfer2'
folder3 = './test_content_resized1'

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

    path1 = os.path.join(folder2, filename)
    path2 = os.path.join(folder3, filename)

    if not os.path.exists(path1) or not os.path.exists(path2):
        print(f"file {filename} doesn't exist, skip...")
        continue

    img1 = np.array(Image.open(path1).convert("RGB"))
    img2 = np.array(Image.open(path2).convert("RGB"))

    if img1.shape != img2.shape:
        # Resize img2 to match the shape of img1
        img2_resized = Image.fromarray(img2).resize(img1.shape[1::-1])  # img1.shape[1::-1] gets the (width, height)
        img2 = np.array(img2_resized)

    psnr_value = psnr(img1, img2, data_range=img1.max() - img1.min())
    ssim_value = ssim(img1, img2, data_range=img1.max() - img1.min(), multichannel=True)

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
