# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything Demo: Images-Only Inference with Visualization

Usage:
    python demo_images_only_inference.py --help
"""

import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import rerun as rr
import torch

from mapanything.models import MapAnything
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.image import load_images
from mapanything.utils.inference import add_global_semantic_pca_rgb, compute_global_semantic_clusters
from mapanything.utils.viz import (
    predictions_to_glb,
    script_add_rerun_args,
)
from typing import Tuple

def load_encoder_checkpoint(
    model_without_ddp,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[int, float]:
    """
    Load checkpoint containing only ENCODER trainable components.
    
    Loads:
        - dpt_feature_head_2 (student encoder)
        - sam2_compat (student encoder compatibility layer)
        - unfrozen info_sharing blocks (multi-view transformer)
        - unfrozen DINOv2 encoder blocks
    
    Args:
        model_without_ddp: Model without DDP wrapper
        checkpoint_path: Path to encoder checkpoint
        device: Device to load checkpoint on
        args: Training arguments (optional, not required for loading)
    
    Returns:
        (start_epoch, best_val_loss) tuple from checkpoint
    """
    print(f"[LOAD] Loading encoder checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"[DEBUG] Checkpoint content: {ckpt.keys()}")
    
    # Load encoder head
    if "dpt_feature_head_2" in ckpt:
        model_without_ddp.dpt_feature_head_2.load_state_dict(ckpt["dpt_feature_head_2"])
        print("[INFO] Loaded dpt_feature_head_2")
    else:
        print("[WARN] dpt_feature_head_2 not found in encoder checkpoint!")
    
    # Load sam2_compat if present
    if "sam2_compat" in ckpt and hasattr(model_without_ddp, "sam2_compat"):
        model_without_ddp.sam2_compat.load_state_dict(ckpt["sam2_compat"])
        print("[INFO] Loaded sam2_compat")
    elif hasattr(model_without_ddp, "sam2_compat"):
        print("[INFO] sam2_compat not in encoder checkpoint; using random initialization")
    
    # Load info_sharing blocks (NO dipendenza da args)
    if "info_sharing_blocks" in ckpt and hasattr(model_without_ddp, "info_sharing"):
        info = model_without_ddp.info_sharing
        blocks = getattr(info, "self_attention_blocks", None)
        if blocks:
            for idx, state_dict in ckpt["info_sharing_blocks"].items():
                if idx < len(blocks):
                    try:
                        blocks[idx].load_state_dict(state_dict)
                    except Exception as e:
                        print(f"[WARN] Failed loading info_sharing block {idx}: {e}")
            print(f"[INFO] Restored {len(ckpt['info_sharing_blocks'])} info_sharing blocks")
        
        if "info_sharing_wrappers" in ckpt:
            for name, data in ckpt["info_sharing_wrappers"].items():
                try:
                    param = dict(info.named_parameters())[name]
                    param.data.copy_(data)
                except Exception as e:
                    print(f"[WARN] Failed loading wrapper param {name}: {e}")
            print(f"[INFO] Restored {len(ckpt['info_sharing_wrappers'])} info_sharing wrapper params")
    
    # Load DINOv2 blocks (NO dipendenza da args)
    if "dino_encoder_blocks" in ckpt and hasattr(model_without_ddp, "encoder"):
        dino_encoder = model_without_ddp.encoder
        dino_model = dino_encoder.model if hasattr(dino_encoder, "model") else dino_encoder
        blocks = getattr(dino_model, "blocks", None)
        if blocks:
            for idx, state_dict in ckpt["dino_encoder_blocks"].items():
                if idx < len(blocks):
                    try:
                        blocks[idx].load_state_dict(state_dict)
                    except Exception as e:
                        print(f"[WARN] Failed loading DINOv2 block {idx}: {e}")
            print(f"[INFO] Restored {len(ckpt['dino_encoder_blocks'])} DINOv2 blocks")
        
        if "dino_encoder_wrappers" in ckpt:
            for name, data in ckpt["dino_encoder_wrappers"].items():
                try:
                    param = dict(dino_model.named_parameters())[name]
                    param.data.copy_(data)
                except Exception as e:
                    print(f"[WARN] Failed loading wrapper param {name}: {e}")
            print(f"[INFO] Restored {len(ckpt['dino_encoder_wrappers'])} DINOv2 wrapper params")
    
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    
    return start_epoch, best_val_loss


def log_data_to_rerun(
    image,
    depthmap,
    pose,
    intrinsics,
    pts3d,
    mask,
    base_name,
    view_idx,
    viz_mask=None,
    semantic_pca_rgb=None,
    semantic_clusters_local=None,
    semantic_clusters_global=None,
    log_semantic_pointcloud=False,
):
    """Log visualization data to Rerun"""
    # Log camera info and loaded data
    height, width = image.shape[0], image.shape[1]
    scene_root = "mapanything/mapanything"
    pointcloud_root = f"{scene_root}/pointclouds"
    semantic_pca_root = f"{scene_root}/semantic_pca"
    cluster_local_root = f"{scene_root}/cluster_local"
    cluster_global_root = f"{scene_root}/cluster_global"

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

    # Log semantic visualizations if available
    if semantic_pca_rgb is not None:
        rr.log(
            f"{semantic_pca_root}/semantic_{view_idx}",
            rr.Image(semantic_pca_rgb),
        )

    if semantic_clusters_local is not None:
        rr.log(
            f"{cluster_local_root}/cluster_local_{view_idx}",
            rr.SegmentationImage(semantic_clusters_local.astype(np.int32)),
        )

    if semantic_clusters_global is not None:
        rr.log(
            f"{cluster_global_root}/cluster_global_{view_idx}",
            rr.SegmentationImage(semantic_clusters_global.astype(np.int32)),
        )

    # Log points in 3D with RGB colors
    filtered_pts = pts3d[mask]
    filtered_pts_col = image[mask]

    rr.log(
        f"{pointcloud_root}/pointcloud_view_{view_idx}",
        rr.Points3D(
            positions=filtered_pts.reshape(-1, 3),
            colors=filtered_pts_col.reshape(-1, 3),
        ),
    )

    # Log semantic point clouds if requested
    if log_semantic_pointcloud:
        if semantic_pca_rgb is not None:
            filtered_semantic_pca = semantic_pca_rgb[mask]
            rr.log(
                f"{semantic_pca_root}/semantic_{view_idx}/pointcloud",
                rr.Points3D(
                    positions=filtered_pts.reshape(-1, 3),
                    colors=filtered_semantic_pca.reshape(-1, 3),
                ),
            )

        if semantic_clusters_local is not None:
            filtered_clusters = semantic_clusters_local[mask]
            # Convert cluster IDs to colors
            unique_labels = np.unique(filtered_clusters)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

            # Simple colormap for clusters
            from matplotlib import cm
            cmap = cm.get_cmap("tab20", n_clusters + 1)

            cluster_colors = np.zeros((len(filtered_clusters), 3))
            for label in unique_labels:
                cluster_mask = filtered_clusters == label
                if label == -1:
                    cluster_colors[cluster_mask] = [0.5, 0.5, 0.5]  # Gray for noise
                else:
                    color_idx = label % 20
                    cluster_colors[cluster_mask] = cmap(color_idx)[:3]

            rr.log(
                f"{cluster_local_root}/cluster_local_{view_idx}/pointcloud",
                rr.Points3D(
                    positions=filtered_pts.reshape(-1, 3),
                    colors=cluster_colors,
                ),
            )

        if semantic_clusters_global is not None:
            filtered_clusters = semantic_clusters_global[mask]
            unique_labels = np.unique(filtered_clusters)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

            from matplotlib import cm

            cmap = cm.get_cmap("tab20", max(n_clusters + 1, 1))
            cluster_colors = np.zeros((len(filtered_clusters), 3))
            for label in unique_labels:
                cluster_mask = filtered_clusters == label
                if label == -1:
                    cluster_colors[cluster_mask] = [0.5, 0.5, 0.5]
                else:
                    color_idx = int(label) % 20
                    cluster_colors[cluster_mask] = cmap(color_idx)[:3]

            rr.log(
                f"{cluster_global_root}/cluster_global_{view_idx}/pointcloud",
                rr.Points3D(
                    positions=filtered_pts.reshape(-1, 3),
                    colors=cluster_colors,
                ),
            )


def get_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="MapAnything Demo: Visualize metric 3D reconstruction from images"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to folder containing images for reconstruction",
    )
    parser.add_argument(
        "--number_of_views",
        type=int,
        default=None,
        help="Number of views to process (default: all)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        # default="/scratch2/nico/distillation/output/distillation_loss_full_dataset/checkpoints/checkpoint_encoder_10000.pth",
        default="/scratch2/nico/distillation/output/resume_1_consistency_01/checkpoints/checkpoint_encoder_11000.pth",
        # default="/scratch2/nico/distillation/output/resume_2_consistency_05/checkpoints/checkpoint_encoder_11000.pth",
        # default="/scratch2/nico/distillation/output/resume_3_consistency_1/checkpoints/checkpoint_encoder_11000.pth",
        help="Path to encoder checkpoint (optional, for loading trained encoder weights)",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Square input resolution used for preprocessing",
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        default=False,
        help="Use memory efficient inference for reconstruction (trades off speed)",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed model (facebook/map-anything-apache)",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Enable visualization with Rerun",
    )
    parser.add_argument(
        "--viz_semantic",
        action="store_true",
        default=False,
        help="Enable semantic feature visualization (PCA and clustering)",
    )
    parser.add_argument(
        "--viz_semantic_pointcloud",
        action="store_true",
        default=False,
        help="Enable semantic point cloud visualization (requires --viz_semantic)",
    )
    parser.add_argument(
        "--save_glb",
        action="store_true",
        default=False,
        help="Save reconstruction as GLB file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.glb",
        help="Output path for GLB file (default: output.glb)",
    )
    parser.add_argument(
        "--conf_percentile",
        type=float,
        default=50.0,
        help="Confidence percentile threshold (0-100). Points below threshold are filtered out.",
    )

    return parser


def main():
    # Parser for arguments and Rerun
    parser = get_parser()
    script_add_rerun_args(
        parser
    )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    if args.conf_percentile is not None and not (0.0 <= args.conf_percentile <= 100.0):
        raise ValueError("--conf_percentile must be in [0, 100]")

    # Get inference device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model from HuggingFace
    if args.apache:
        model_name = "facebook/map-anything-apache"
        print("Loading Apache 2.0 licensed MapAnything model...")
    else:
        model_name = "facebook/map-anything"
        print("Loading CC-BY-NC 4.0 licensed MapAnything model...")
    # model = MapAnything.from_pretrained(model_name).to(device)
    model = MapAnything.from_pretrained(
            model_name,
            revision="562de9ff7077addd5780415661c5fb031eb8003e",
            strict=False,
        ).to(device)

    _, _ = load_encoder_checkpoint(
            model_without_ddp=model,
            checkpoint_path=args.checkpoint_path,
            device=device,
        )

    # Load images
    print(f"Loading images from: {args.image_folder}")
    if args.number_of_views is not None:
        print(f"Limiting to first {args.number_of_views} views")
        supported_extensions = (".jpg", ".jpeg", ".png", ".heic", ".heif")
        image_names = sorted(
            name
            for name in os.listdir(args.image_folder)
            if name.lower().endswith(supported_extensions)
        )
        views = [
            os.path.join(args.image_folder, name)
            for name in image_names[: args.number_of_views]
        ]
    else:
        views = args.image_folder

    # views = load_images(views)
    views = load_images(
        views,
        resize_mode="square",
        size=args.input_size,
        norm_type="dinov2",
    )
    print(f"Loaded {len(views)} views")

    # Run model inference
    print("Running inference...")
    outputs = model.infer(
        views, memory_efficient_inference=args.memory_efficient_inference
    )
    print("Inference complete!")

    global_semantic_clusters = None
    if args.viz_semantic:
        print("Adding global semantic PCA basis to outputs (distillation-aligned)...")
        outputs = add_global_semantic_pca_rgb(outputs)
        print("Global semantic PCA computation complete!")
        
        print("Computing scene-level semantic clusters...")
        global_semantic_clusters = compute_global_semantic_clusters(
            outputs,
            conf_percentile=args.conf_percentile,
        )
        print("Scene-level semantic clustering complete!")

    # Prepare lists for GLB export if needed
    world_points_list = []
    images_list = []
    masks_list = []

    # Initialize Rerun if visualization is enabled
    if args.viz:
        print("Starting visualization...")
        viz_string = "MapAnything_Visualization"
        rr.script_setup(args, viz_string)
        rr.set_time("stable_time", sequence=0)
        rr.log("mapanything", rr.ViewCoordinates.RDF, static=True)

    # Loop through the outputs
    for view_idx, pred in enumerate(outputs):
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

        # Apply confidence percentile filtering (same logic as Gradio visualization)
        if args.conf_percentile is not None and "conf" in pred:
            conf_np = pred["conf"][0].squeeze().cpu().numpy()
            finite_conf = conf_np[np.isfinite(conf_np)]
            if finite_conf.size == 0:
                print(
                    f"[WARN] View {view_idx}: no finite confidence values found; skipping confidence filter"
                )
            else:
                conf_threshold = np.nanpercentile(conf_np.reshape(-1), args.conf_percentile)
                conf_mask = np.isfinite(conf_np) & (conf_np >= conf_threshold)
                kept_points = int(conf_mask.sum())
                total_points = int(conf_mask.size)
                print(
                    f"[CONF] View {view_idx}: percentile={args.conf_percentile:.2f}, "
                    f"threshold={conf_threshold:.6f}, kept={kept_points}/{total_points}"
                )
                mask = mask & conf_mask

        pts3d_np = pts3d_computed.cpu().numpy()
        image_np = pred["img_no_norm"][0].cpu().numpy()
        
        # Extract semantic features if available and requested
        semantic_pca_rgb_np = None
        semantic_clusters_local_np = None
        semantic_clusters_global_np = None
        
        if args.viz_semantic:
            # Use global PCA basis (aligned with distillation) if available
            if "semantic_pca_rgb_global" in pred and pred["semantic_pca_rgb_global"] is not None:
                semantic_pca_rgb_np = pred["semantic_pca_rgb_global"].cpu().numpy()
                # Upsample to image resolution (64x64 -> image_h x image_w)
                if semantic_pca_rgb_np.shape[:2] != image_np.shape[:2]:
                    pca_tensor = torch.from_numpy(semantic_pca_rgb_np).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
                    img_h, img_w = image_np.shape[:2]
                    pca_upsampled = torch.nn.functional.interpolate(
                        pca_tensor,
                        size=(img_h, img_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    semantic_pca_rgb_np = pca_upsampled.squeeze(0).permute(1, 2, 0).cpu().numpy()
            elif "semantic_pca_rgb" in pred:
                # Fallback to per-view PCA from model prediction
                semantic_pca_rgb_np = pred["semantic_pca_rgb"][0].cpu().numpy()
            else:
                print(f"Warning: semantic_pca_rgb not available for view {view_idx}")
            
            if "semantic_clusters" in pred:
                semantic_clusters_local_np = pred["semantic_clusters"][0].cpu().numpy()
            else:
                print(f"Warning: semantic_clusters not available in predictions for view {view_idx}")

            if global_semantic_clusters is not None and global_semantic_clusters[view_idx] is not None:
                semantic_clusters_global_np = global_semantic_clusters[view_idx]

        # Store data for GLB export if needed
        if args.save_glb:
            world_points_list.append(pts3d_np)
            images_list.append(image_np)
            masks_list.append(mask)

        # Log to Rerun if visualization is enabled
        if args.viz:
            log_data_to_rerun(
                image=image_np,
                depthmap=depthmap_torch.cpu().numpy(),
                pose=camera_pose_torch.cpu().numpy(),
                intrinsics=intrinsics_torch.cpu().numpy(),
                pts3d=pts3d_np,
                mask=mask,
                base_name=f"mapanything/view_{view_idx}",
                view_idx=view_idx,
                viz_mask=mask,
                semantic_pca_rgb=semantic_pca_rgb_np,
                semantic_clusters_local=semantic_clusters_local_np,
                semantic_clusters_global=semantic_clusters_global_np,
                log_semantic_pointcloud=args.viz_semantic_pointcloud,
            )

    if args.viz:
        print("Visualization complete! Check the Rerun viewer.")
        if args.viz_semantic:
            print("  - Semantic PCA visualization: mapanything/mapanything/semantic_pca/semantic_*")
            print("  - Local semantic clusters: mapanything/mapanything/cluster_local/cluster_local_*")
            print("  - Global semantic clusters: mapanything/mapanything/cluster_global/cluster_global_*")
        if args.viz_semantic_pointcloud:
            print("  - Point clouds: mapanything/mapanything/pointclouds/pointcloud_view_*")
            print("  - Semantic PCA point cloud: mapanything/mapanything/semantic_pca/semantic_*/pointcloud")
            print("  - Local semantic cluster point cloud: mapanything/mapanything/cluster_local/cluster_local_*/pointcloud")
            print("  - Global semantic cluster point cloud: mapanything/mapanything/cluster_global/cluster_global_*/pointcloud")

    # Export GLB if requested
    if args.save_glb:
        print(f"Saving GLB file to: {args.output_path}")

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
        scene_3d.export(args.output_path)
        print(f"Successfully saved GLB file: {args.output_path}")
    else:
        print("Skipping GLB export (--save_glb not specified)")


if __name__ == "__main__":
    main()
