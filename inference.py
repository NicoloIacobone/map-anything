# Optional config for better memory efficiency
import os
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
import torch
import time
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from pathlib import Path
import os

images = ["/scratch2/nico/distillation/dataset/coco2017/images/train2017/000000000030.jpg"]
OUTPUT_DIR = Path("/scratch2/nico/distillation/tests/embeddings/student")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
resume_ckpt = "/scratch2/nico/distillation/output/distillation_3/checkpoints/checkpoint_best.pth"

# Get inference device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load model
model = MapAnything.from_pretrained("facebook/map-anything", strict=False).to(device)
print(f"Model loaded. Has dpt_feature_head_2: {hasattr(model, 'dpt_feature_head_2')}")

# Load weights
ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
# Load DPT feature head weights
model.dpt_feature_head_2.load_state_dict(ckpt["dpt_feature_head_2"])
# Set model to eval mode
model.eval()

# Load and preprocess images from a folder or list of paths
print(f"Loading images from: {images}")
views = load_images(images)
print(f"Loaded {len(views)} views")

# Set dtype for autocasting
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

for v in views:
    v["img"] = v["img"].to(device=device, dtype=dtype, non_blocking=True)

# Run inference
print("Running inference...")
start_time = time.time()

# Inference with autocasting for better performance
with torch.autocast(device_type="cuda", enabled=True, dtype=dtype):
    predictions = model(
        views,
        memory_efficient_inference=False,
    )

# Extract and save student features
student_features = getattr(model, "_last_feat2_8x", None)
torch.save(student_features.detach().cpu(), OUTPUT_DIR / f"000000000030.pt")
print(f"Saved student features to {OUTPUT_DIR / f'000000000030.pt'}")

elapsed_time = time.time() - start_time
print(f"Inference complete! Elapsed time: {elapsed_time:.2f} seconds")

# predictions = model.infer(
#     views,
#     memory_efficient_inference=False,
#     use_amp=True,
#     amp_dtype="bf16",
#     apply_mask=True,
#     mask_edges=True,
#     apply_confidence_mask=False,
#     confidence_percentile=0,
# )

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
