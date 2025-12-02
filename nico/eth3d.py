import numpy as np
import os
import matplotlib.pyplot as plt

DATA_PATH = "/scratch2/nico/distillation/dataset/ETH3D/npy/0.25"
SCENE_NAME = "courtyard_images.npy"

scene_file = os.path.join(DATA_PATH, SCENE_NAME)
images = np.load(scene_file)  # shape: (N, H, W, C) or (N, H, W)

# Create output directory based on SCENE_NAME (without extension)
output_dir = os.path.join(DATA_PATH, os.path.splitext(SCENE_NAME)[0])
os.makedirs(output_dir, exist_ok=True)

for i, img in enumerate(images):
    # If RGB channels-first, transpose to channels-last
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))  # (3, H, W) → (H, W, 3)
    # Save image as PNG
    plt.imsave(os.path.join(output_dir, f"image_{i:04d}.png"), img, cmap='gray' if img.ndim != 3 or img.shape[2] != 3 else None)
