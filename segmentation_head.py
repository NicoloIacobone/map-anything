# Optional config for better memory efficiency
import os
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
import torch
import time
import numpy as np
import rerun as rr
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.geometry import depthmap_to_world_frame
import os
from mapanything.utils.viz import (
    predictions_to_glb,
    script_add_rerun_args,
)

def log_data_to_rerun(
    image, depthmap, pose, intrinsics, pts3d, mask, base_name, pts_name, viz_mask=None
):
    """Log visualization data to Rerun"""
    # Log camera info and loaded data
    height, width = image.shape[0], image.shape[1]
    rr.log(
        base_name,
        rr.Transform3D(
            translation=pose[:3, 3],
            mat3x3=pose[:3, :3],
        ),
    )
    rr.log(
        f"{base_name}/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=1.0,
        ),
    )
    rr.log(
        f"{base_name}/pinhole/rgb",
        rr.Image(image),
    )
    rr.log(
        f"{base_name}/pinhole/depth",
        rr.DepthImage(depthmap),
    )
    if viz_mask is not None:
        rr.log(
            f"{base_name}/pinhole/mask",
            rr.SegmentationImage(viz_mask.astype(int)),
        )

    # Log points in 3D
    filtered_pts = pts3d[mask]
    filtered_pts_col = image[mask]

    rr.log(
        pts_name,
        rr.Points3D(
            positions=filtered_pts.reshape(-1, 3),
            colors=filtered_pts_col.reshape(-1, 3),
        ),
    )

def get_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="MapAnything Demo: Visualize metric 3D reconstruction from images"
    )

    return parser

# Arguments
parser = get_parser()
script_add_rerun_args(
    parser
)  # Options: --headless, --connect, --serve, --addr, --save, --stdout
args = parser.parse_args()

save_glb = False  # Whether to save the output as a GLB file
input_path = "/scratch2/nico/examples/photos"
output_base_path = "/scratch2/nico/distillation/mapanything"
folder_names = ["box_ufficio", "yokohama", "tenda_ufficio", "sedia_ufficio", "pianta", "car_drift"]

load_weights = True
ckpt_path = "/scratch2/nico/distillation/output/checkpoint_final.pth"

# Get inference device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MapAnything.from_pretrained("facebook/map-anything").to(device)

if load_weights:
    checkpoint = torch.load(ckpt_path, map_location=device) # carica su device
    model.instance_head.load_state_dict(checkpoint["instance_head"]) # carica solo head
    output_base_path = os.path.join(output_base_path, "distilled")
    print(f"[DEBUG] Loaded weights for additional head from {ckpt_path}")
else:
    output_base_path = os.path.join(output_base_path, "not_distilled")
    print("[DEBUG] Using not trained additional head")

for folder_name in folder_names:
    output_path = os.path.join(output_base_path, folder_name)
    images = os.path.join(input_path, folder_name)

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

    print('[DEBUG] Infer OK. Num views:', len(predictions))

    # Save the last instance embeddings in a folder
    embeddings = getattr(model, "_last_inst_embeddings", None)
    if embeddings is not None:
        embeddings_path = os.path.join(output_path, "student_embeddings.pt")
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        torch.save(embeddings.cpu(), embeddings_path)
        print(f"[DEBUG] Saved instance embeddings to {embeddings_path}")
    else:
        print("[DEBUG] No instance embeddings found to save.")

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