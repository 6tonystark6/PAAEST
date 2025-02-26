import os
import glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io, transform
from tqdm import tqdm

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.ToTensor(),
    normalize
])

def denorm(tensor, device):
    tensor = tensor.to(device)
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

class PreprocessDataset(Dataset):
    def __init__(self, content_dir, style_dir, transforms=trans):
        content_dir_resized = content_dir + '_resized'
        style_dir_resized = style_dir + '_resized'
        if not (os.path.exists(content_dir_resized) and
                os.path.exists(style_dir_resized)):
            os.makedirs(content_dir_resized, exist_ok=True)
            os.makedirs(style_dir_resized, exist_ok=True)
            self._resize(content_dir, content_dir_resized)
            self._resize(style_dir, style_dir_resized)
        content_images = glob.glob(os.path.join(content_dir_resized, '*'))
        np.random.shuffle(content_images)
        style_images = glob.glob(os.path.join(style_dir_resized, '*'))
        np.random.shuffle(style_images)
        self.images_pairs = list(zip(content_images, style_images))
        self.transforms = transforms

    @staticmethod
    def _resize(source_dir, target_dir):
        print(f'Start resizing {source_dir} ')
        for i in tqdm(os.listdir(source_dir)):
            filename = os.path.basename(i)
            try:
                image_path = os.path.join(source_dir, i)
                image = io.imread(image_path)
                if len(image.shape) == 3 and image.shape[-1] == 3:
                    H, W, _ = image.shape
                    if H < W:
                        ratio = W / H
                        H = 512
                        W = int(ratio * H)
                    else:
                        ratio = H / W
                        W = 512
                        H = int(ratio * W)
                    image = transform.resize(image, (H, W), mode='reflect', anti_aliasing=True)
                    image = (image * 255).astype(np.uint8)
                    io.imsave(os.path.join(target_dir, filename), image)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    def __len__(self):
        return len(self.images_pairs)

    def __getitem__(self, index):
        content_image, style_image = self.images_pairs[index]
        content_image = Image.open(content_image)
        style_image = Image.open(style_image)
        if self.transforms:
            content_image = self.transforms(content_image)
            style_image = self.transforms(style_image)
        return content_image, style_image
