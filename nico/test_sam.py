import numpy as np
import os
from feature_extractor import load_sam2_feature_extractor
import torch

from PIL import Image

DATA_PATH = "/scratch2/nico/distillation/dataset/ETH3D/npy/0.25/courtyard_images"
output_dir = os.path.join(DATA_PATH, "features")
os.makedirs(output_dir, exist_ok=True)
images = [os.path.join(DATA_PATH, fname) for fname in os.listdir(DATA_PATH) if fname.endswith(".png")]

# Setup teacher
teacher = load_sam2_feature_extractor(
    checkpoint_path="/scratch2/nico/sam2/checkpoints/sam2.1_hiera_large.pt",
    device="cuda"
)

for image in images:
    pil_img = Image.open(image).convert("RGB")
    teacher_features = teacher(pil_img)  # (B, 256, 64, 64)
    image_name = os.path.splitext(os.path.basename(image))[0]
    output_path = os.path.join(output_dir, f"features_{image_name}.pt")
    torch.save(torch.tensor(teacher_features), output_path)
    print("Saved features to:", output_path, "with shape:", teacher_features.shape)