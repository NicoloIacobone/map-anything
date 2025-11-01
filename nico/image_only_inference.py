# Optional config for better memory efficiency
import os
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
import torch
import time
import cv2
import numpy as np
import rerun as rr
import matplotlib.pyplot as plt
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.viz import (
    predictions_to_glb,
    script_add_rerun_args,
)

def build_semantic_map(masks):
    H, W = masks[0]['segmentation'].shape
    seg_map = np.zeros((H, W), dtype=np.int32)
    for idx, m in enumerate(masks, start=1):  # id=0 sarà background
        mask = m['segmentation']
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]  # Rimuovi dimensione batch se presente
        score = m.get('predicted_iou', 1.0)
        if score > 0.8:  # opzionale
            label = int(m.get('class_id', idx))  # usa class_id se fornito, altrimenti idx
            seg_map[mask.astype(bool)] = label
    return seg_map

def colormap_from_segmentation(seg_map):
    """Convert segmentation map (H, W) to RGB color map."""
    cmap = plt.get_cmap("tab20")
    max_id = int(seg_map.max())
    colors = cmap(np.linspace(0, 1, max_id + 1))[:, :3]
    return colors[seg_map]

def log_data_to_rerun(
    image, depthmap, pose, intrinsics, pts3d, mask, base_name, pts_name, viz_mask=None, semantic_colors=None
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
    # Log semantic overlay as a separate layer if provided
    if semantic_colors is not None:
        rr.log(
            f"{base_name}/pinhole/semantic_overlay",
            rr.Image(semantic_colors),
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

    # === Semantic rendering section ===
    # Log points in 3D with RGB colors
    filtered_pts = pts3d[mask]
    filtered_pts_col_rgb = image[mask]
    rr.log(
        pts_name,
        rr.Points3D(
            positions=filtered_pts.reshape(-1, 3),
            colors=filtered_pts_col_rgb.reshape(-1, 3),
        ),
    )
    # Log the semantic overlay as an additional 3D points layer if provided
    if semantic_colors is not None:
        rr.log(
            f"{pts_name}_semantic_overlay",
            rr.Points3D(
                positions=filtered_pts.reshape(-1, 3),
                colors=semantic_colors[mask].reshape(-1, 3),
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
images = "/scratch2/nico/examples/photos/box_ufficio"
output_path = "/scratch2/nico/examples/photos/results"

# Get inference device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init model - This requries internet access or the huggingface hub cache to be pre-downloaded
# For Apache 2.0 license model, use "facebook/map-anything-apache"
model = MapAnything.from_pretrained("facebook/map-anything").to(device)

# Load and preprocess images from a folder or list of paths
print(f"Loading images from: {images}")
views = load_images(images)
print(f"Loaded {len(views)} views")

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
print(f"Inference complete! Elapsed time: {elapsed_time:.2f} seconds")

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

    # === Semantic rendering section ===
    # === Load precomputed SAM segmentation masks ===
    seg_map_resized = None
    seg_colors = None
    masks_path = os.path.join(images, "masks.npy")
    if os.path.exists(masks_path):
        masks_data = np.load(masks_path, allow_pickle=True)
        masks_for_view = None
        
        if isinstance(masks_data, np.ndarray) and len(masks_data) > 0:
            # Tipologia 1: lista di dict (single-view o multi-view con lista)
            if isinstance(masks_data[0], dict) and 'segmentation' in masks_data[0]:
                # Formato già compatibile (lista di dict)
                masks_for_view = masks_data.tolist() if isinstance(masks_data, np.ndarray) else masks_data
            elif isinstance(masks_data[0], dict) and 'segmentation' not in masks_data[0]:
                # Tipologia 2: dizionario di dizionari {frame_idx: {obj_id: mask}}
                # Converti in formato compatibile per il frame corrente
                frame_data = masks_data.item() if masks_data.dtype == object else masks_data
                if view_idx in frame_data or str(view_idx) in frame_data:
                    frame_masks = frame_data.get(view_idx) or frame_data.get(str(view_idx))
                    if isinstance(frame_masks, dict):
                        # Converti {obj_id: mask_array} in lista di dict
                        masks_for_view = []
                        for obj_id, mask_arr in frame_masks.items():
                            m = np.asarray(mask_arr)
                            if m.ndim == 3 and m.shape[0] == 1:
                                m = m[0]
                            m = m.astype(bool)
                            masks_for_view.append({
                                'segmentation': m,
                                'predicted_iou': 1.0,
                                'class_id': int(obj_id)
                            })
        
        # Se abbiamo trovato maschere per questa view, procedi
        if masks_for_view and len(masks_for_view) > 0:
            seg_map = build_semantic_map(masks_for_view)

            # Resize SAM segmentation to match MapAnything resolution
            H_map, W_map = image_np.shape[:2]
            seg_map_resized = cv2.resize(
                seg_map.astype(np.uint8),
                (W_map, H_map),
                interpolation=cv2.INTER_NEAREST,  # Preserve discrete labels
            )

            # Convert segmentation IDs to colors (resized version)
            seg_colors = colormap_from_segmentation(seg_map_resized)
            seg_colors = np.clip(seg_colors, 0, 1)

            # Log the 2D segmentation image to Rerun for reference
            rr.log(f"mapanything/view_{view_idx}/pinhole/semantic_segmentation",
                rr.SegmentationImage(seg_map_resized))

    # Store data for GLB export if needed
    if save_glb:
        # Prepare lists for GLB export if needed
        world_points_list = []
        images_list = []
        masks_list = []
        
        world_points_list.append(pts3d_np)
        images_list.append(image_np)
        masks_list.append(mask)

    # Log to Rerun for visualization with semantic colors
    log_data_to_rerun(
        image=image_np,
        depthmap=depthmap_torch.cpu().numpy(),
        pose=camera_pose_torch.cpu().numpy(),
        intrinsics=intrinsics_torch.cpu().numpy(),
        pts3d=pts3d_np,
        mask=mask,
        base_name=f"mapanything/view_{view_idx}",
        pts_name=f"mapanything/pointcloud_view_{view_idx}_semantic",
        viz_mask=seg_map_resized,
        semantic_colors=seg_colors
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