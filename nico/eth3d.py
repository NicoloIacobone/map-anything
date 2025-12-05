import numpy as np
import os
import matplotlib.pyplot as plt

DATA_PATH = "/scratch2/nico/distillation/dataset/ETH3D/images/train"

for fname in sorted(os.listdir(DATA_PATH)):
    if not fname.endswith(".npy"):
        continue
    scene_file = os.path.join(DATA_PATH, fname)
    try:
        images = np.load(scene_file)
    except Exception as e:
        print(f"Skipping {fname}: load error: {e}")
        continue

    # Normalize shapes to a batch of images with channels-last:
    # (N,3,H,W) -> (N,H,W,3)
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    elif images.ndim == 3:
        # single image in channels-first (3,H,W) -> (H,W,3) then add batch dim
        if images.shape[0] == 3:
            images = np.transpose(images, (1, 2, 0))[np.newaxis, ...]
        # single image in channels-last (H,W,C) -> add batch dim
        elif images.shape[2] == 3 or images.shape[2] == 1:
            images = images[np.newaxis, ...]
        # otherwise assume (N, H, W) and leave as-is
    elif images.ndim == 2:
        # single grayscale image (H, W) -> add batch dim
        images = images[np.newaxis, ...]

    output_dir = os.path.join(DATA_PATH, os.path.splitext(fname)[0])
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(images):
        # If RGB channels-first per-image (3, H, W), transpose
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        cmap = "gray" if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1) else None
        plt.imsave(os.path.join(output_dir, f"image_{i:04d}.png"), img, cmap=cmap)
