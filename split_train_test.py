import os
import shutil
import random

data_dir = "./content_data"
train_dir = "./train_content"
test_dir = "./test_content"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

images = [f for f in os.listdir(data_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

random.shuffle(images)

split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
test_images = images[split_idx:]

for idx, img in enumerate(train_images, start=1):
    shutil.copy(os.path.join(data_dir, img), os.path.join(train_dir, f"{idx}.png"))

for idx, img in enumerate(test_images, start=1):
    shutil.copy(os.path.join(data_dir, img), os.path.join(test_dir, f"{idx}.png"))

print("train test split complete")
