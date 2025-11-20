# Optional config for better memory efficiency
import os
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
import torch
import time
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

import os

images = ["/scratch2/nico/distillation/dataset/coco2017/images/val2017/000000003553.jpg"]
# images = "/scratch2/nico/examples/photos/test_features"

# Get inference device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MapAnything.from_pretrained("facebook/map-anything", strict=False).to(device)

# Load and preprocess images from a folder or list of paths
print(f"Loading images from: {images}")
views = load_images(images)
print(f"Loaded {len(views)} views")

# Run inference
print("Running inference...")
start_time = time.time()
predictions = model.infer(
    views,
    memory_efficient_inference=False,
    use_amp=True,
    amp_dtype="bf16",
    apply_mask=True,
    mask_edges=True,
    apply_confidence_mask=False,
    confidence_percentile=0,
)
elapsed_time = time.time() - start_time
print(f"Inference complete! Elapsed time: {elapsed_time:.2f} seconds")

# Access results for each view - Complete list of metric outputs
# for i, pred in enumerate(predictions):
#     # Geometry outputs
#     pts3d = pred["pts3d"]                     # 3D points in world coordinates (B, H, W, 3)
#     pts3d_cam = pred["pts3d_cam"]             # 3D points in camera coordinates (B, H, W, 3)
#     depth_z = pred["depth_z"]                 # Z-depth in camera frame (B, H, W, 1)
#     depth_along_ray = pred["depth_along_ray"] # Depth along ray in camera frame (B, H, W, 1)

#     # Camera outputs
#     ray_directions = pred["ray_directions"]   # Ray directions in camera frame (B, H, W, 3)
#     intrinsics = pred["intrinsics"]           # Recovered pinhole camera intrinsics (B, 3, 3)
#     camera_poses = pred["camera_poses"]       # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world poses in world frame (B, 4, 4)
#     cam_trans = pred["cam_trans"]             # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world translation in world frame (B, 3)
#     cam_quats = pred["cam_quats"]             # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world quaternion in world frame (B, 4)

#     # Quality and masking
#     confidence = pred["conf"]                 # Per-pixel confidence scores (B, H, W)
#     mask = pred["mask"]                       # Combined validity mask (B, H, W, 1)
#     non_ambiguous_mask = pred["non_ambiguous_mask"]                # Non-ambiguous regions (B, H, W)
#     non_ambiguous_mask_logits = pred["non_ambiguous_mask_logits"]  # Mask logits (B, H, W)

#     # Scaling
#     metric_scaling_factor = pred["metric_scaling_factor"]  # Applied metric scaling (B,)

#     # Original input
#     img_no_norm = pred["img_no_norm"]         # Denormalized input images for visualization (B, H, W, 3)
