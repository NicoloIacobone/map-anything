# rerun --serve-web --port 2004 --web-viewer-port 2006

# Optional config for better memory efficiency
import os
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
from matplotlib import image
import torch
import time
import cv2
import numpy as np
import glob
from PIL import Image
import rerun as rr
import matplotlib.pyplot as plt
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.viz import (
    predictions_to_glb,
    script_add_rerun_args,
)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

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
images = "/scratch2/nico/examples/photos/test_multi_view_single_inference"
# images = "/scratch2/nico/examples/photos/single_frame"
output_path = "/scratch2/nico/examples/photos/results"

masks_path = os.path.join(images, "masks.npy")
masks_data = np.load(masks_path, allow_pickle=True)

frame_paths = sorted(glob.glob(os.path.join(images, "*.jpg")))
frames = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]

plt.figure(figsize=(20, 20))
plt.imshow(frames[1])
show_anns(masks_data[1])

plt.show()



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