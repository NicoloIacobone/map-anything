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
images = "/scratch2/nico/examples/photos/pianta"
# images = "/scratch2/nico/sam2/notebooks/images/cars"
output_path = "/scratch2/nico/examples/photos/results"

# Get inference device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init model - This requries internet access or the huggingface hub cache to be pre-downloaded
# For Apache 2.0 license model, use "facebook/map-anything-apache"
model = MapAnything.from_pretrained("facebook/map-anything").to(device)

# Load and preprocess images from a folder or list of paths
print(f"Loading images from: {images}")
views = load_images(images)
# views = load_images(images, resize_mode="square", size=518, verbose=True)
print(f"Loaded {len(views)} views")
print(f"Image shape before preprocessing: {views[0]['img'].shape}, dtype: {views[0]['img'].dtype}")
import matplotlib.pyplot as plt

first_img_norm = views[0]['img']            # (1,3,H,W) tensor normalizzato
tensor_chw = first_img_norm[0]              # (3,H,W)
img_hwc = tensor_chw.permute(1, 2, 0).cpu() # (H,W,3)

# Denormalizza (dinov2) se vuoi colori corretti
from mapanything.utils.image import IMAGE_NORMALIZATION_DICT
stats = IMAGE_NORMALIZATION_DICT['dinov2']
img_vis = img_hwc * stats.std.view(1,1,3) + stats.mean.view(1,1,3)
img_vis = img_vis.clamp(0,1)
plt.imshow(img_vis.numpy())
plt.title("Prima immagine caricata")
plt.axis('off')
plt.show()

# Run inference
print("Running inference...")
start_time = time.time()
predictions = model.infer(
    views,                            # Input views
    memory_efficient_inference=False, # Trades off speed for more views (up to 2000 views on 140 GB)
    use_amp=True,                     # Use mixed precision inference (recommended)
    amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
    apply_mask=True,                  # Apply masking to dense geometry outputs
    mask_edges=True,                  # Remove edge artifacts by using normals and depth
    apply_confidence_mask=False,      # Filter low-confidence regions
    confidence_percentile=0,         # Remove bottom 10 percentile confidence pixels
)
elapsed_time = time.time() - start_time
print(f"[debug] Inference complete! Elapsed time: {elapsed_time:.2f} seconds")

# creation of shopping list for pipeline of distillation/inference
shopping_list = {
    "pointmap": [],         # per-scene 3D pointmap (MapAnything)                                   (B, H, W, 3) es. (7, 336, 518, 3)
    "point_embeddings": [], # per-point embeddings                                                  (B, H, W, C) es. (7, 192, 256, 256) <-- check upsampling or downsampling of pts3d                  
    "correspondences": [],  # per-point 3D-2D correspondences (for each frame in which it's visible)(N, V, 3)    es. (123456, 7, 3)     <-- for each 3D point, the 2D pixel coords + depth in each view (or -1 if not visible)
    "masks": [],            # per-frame 2D segmentation masks (SAM2)                                (B, H, W, 1) es. (7, 336, 518, 1)
    "mask_embeddings": [],  # per-mask embeddings                                                   (B, H, W, C) es. (7, ?, ?, ?)       <-- check SAM2 output
    "poses": []             # camera poses (for reprojection purposes)                              (B, 4, 4)    es. (7, 4, 4)          OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world poses in world frame
}

shopping_list["pointmap"] = [pred["pts3d"].cpu().numpy() for pred in predictions] # forse anche .detach()

# Prepare lists for GLB export if needed
world_points_list = []
images_list = []
masks_list = []

print("Starting visualization...")
viz_string = "MapAnything_Visualization"
rr.script_setup(args, viz_string)
rr.set_time("stable_time", sequence=0)
rr.log("mapanything", rr.ViewCoordinates.RDF, static=True)

for view_idx, pred in enumerate(predictions):
    # Extract data from predictions
    depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
    intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
    camera_pose_torch = pred["camera_poses"][0]  # (4, 4)

    # Compute new pts3d using depth, intrinsics, and camera pose
    pts3d_computed, valid_mask = depthmap_to_world_frame(
        depthmap_torch, intrinsics_torch, camera_pose_torch
    )

    # Convert to numpy arrays
    mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
    mask = mask & valid_mask.cpu().numpy()  # Combine with valid depth mask
    pts3d_np = pts3d_computed.cpu().numpy()
    image_np = pred["img_no_norm"][0].cpu().numpy()

    # Store data for GLB export if needed
    if save_glb:
        world_points_list.append(pts3d_np)
        images_list.append(image_np)
        masks_list.append(mask)

    # Log to Rerun for visualization
    log_data_to_rerun(
        image=image_np,
        depthmap=depthmap_torch.cpu().numpy(),
        pose=camera_pose_torch.cpu().numpy(),
        intrinsics=intrinsics_torch.cpu().numpy(),
        pts3d=pts3d_np,
        mask=mask,
        base_name=f"mapanything/view_{view_idx}",
        pts_name=f"mapanything/pointcloud_view_{view_idx}",
        viz_mask=mask,
    )

print("Visualization complete! Check the Rerun viewer.")

# Export GLB if requested
if save_glb:
    print(f"Saving GLB file to: {output_path}")

    # Stack all views
    world_points = np.stack(world_points_list, axis=0)
    images = np.stack(images_list, axis=0)
    final_masks = np.stack(masks_list, axis=0)

    # Create predictions dict for GLB export
    predictions = {
        "world_points": world_points,
        "images": images,
        "final_masks": final_masks,
    }

    # Convert to GLB scene
    scene_3d = predictions_to_glb(predictions, as_mesh=True)

    # Save GLB file
    scene_3d.export(output_path)
    print(f"Successfully saved GLB file: {output_path}")
else:
    print("Skipping GLB export.")

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