# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Inference utilities.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from mapanything.utils.geometry import (
    depth_edge,
    get_rays_in_camera_frame,
    normals_edge,
    points_to_normals,
    quaternion_to_rotation_matrix,
    recover_pinhole_intrinsics_from_ray_directions,
    rotation_matrix_to_quaternion,
    depthmap_to_world_frame
)
from mapanything.utils.image import rgb
from nico.utils import pca_visualization, pca_visualization_student_only


# Import for semantic feature processing
try:
    from sklearn.decomposition import PCA
    import hdbscan
    SKLEARN_AVAILABLE = True
    warnings.filterwarnings(
        "ignore",
        message=".*force_all_finite.*ensure_all_finite.*",
        category=FutureWarning,
    )
    # Keep clustering tractable at inference time.
    MAX_HDBSCAN_POINTS = 30000
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn and/or hdbscan not available. Semantic feature processing will be disabled.")

# Hard constraints - exactly what users can provide
ALLOWED_VIEW_KEYS = {
    "img",  # Required - input images
    "data_norm_type",  # Required - normalization type of the input images
    "depth_z",  # Optional - Z depth maps
    "ray_directions",  # Optional - ray directions in camera frame
    "intrinsics",  # Optional - pinhole camera intrinsics (conflicts with ray_directions)
    "camera_poses",  # Optional - camera poses
    "is_metric_scale",  # Optional - whether inputs are metric scale
    "true_shape",  # Optional - original image shape
    "idx",  # Optional - index of the view
    "instance",  # Optional - instance info of the view
}

REQUIRED_KEYS = {"img", "data_norm_type"}

# Define conflicting keys that cannot be used together
CONFLICTING_KEYS = [
    ("intrinsics", "ray_directions")  # Both represent camera projection
]

def loss_of_one_batch_multi_view(
    batch,
    model,
    criterion,
    device,
    use_amp=False,
    amp_dtype="bf16",
    ret=None,
    ignore_keys=None,
    teacher_features=None,
    save_pca_visualization_path=None,
    epoch=None,
):
    """
    Calculate loss for a batch with multiple views.

    Args:
        batch (list): List of view dictionaries containing input data.
        model (torch.nn.Module): Model to run inference with.
        criterion (callable, optional): Loss function to compute the loss.
        device (torch.device): Device to run the computation on.
        use_amp (bool, optional): Whether to use automatic mixed precision. Defaults to False.
        amp_dtype (str, optional): Floating point type to use for automatic mixed precision. Options: ["fp32", "fp16", "bf16"]. Defaults to "bf16".
        ret (str, optional): If provided, return only the specified key from the result dictionary.
        ignore_keys (set, optional): Set of keys to ignore when moving tensors to device.
                                   Defaults to {"dataset", "label", "instance",
                                   "idx", "true_shape", "rng", "data_norm_type"}.

    Returns:
        dict or Any: If ret is None, returns a dictionary containing views, predictions, and loss.
                     Otherwise, returns the value associated with the ret key.
    """
    # Move necessary tensors to device
    if ignore_keys is None:
        ignore_keys = set(
            [
                "depthmap",
                "dataset",
                "label",
                "instance",
                "idx",
                "true_shape",
                "rng",
                "data_norm_type",
            ]
        )
    for view in batch:
        for name in view.keys():
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    # Determine the mixed precision floating point type
    if use_amp:
        if amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif amp_dtype == "bf16":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                warnings.warn(
                    "bf16 is not supported on this device. Using fp16 instead."
                )
                amp_dtype = torch.float16
        elif amp_dtype == "fp32":
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    # Run model and compute loss
    with torch.autocast("cuda", enabled=bool(use_amp), dtype=amp_dtype):
        preds = model(batch)
        # ========= EXTRACT STUDENT FEATURES AND CONFIDENCE IF AVAILABLE =========
        # 1. Ottieni PRIMA il modello base (gestisce sia Single-GPU che DDP)
        base_model = model.module if hasattr(model, "module") else model
        if hasattr(base_model, "dpt_feature_head_2"):
            # Estrai le feature (256 canali)
            student_features = getattr(base_model, "_last_feat2_8x", None)
            # for i in range(student_features.shape[0]):
                # pca_visualization(student_features[i], student_features[i])
            
            # Estrai la confidenza appresa (1 canale, softplus attivata)
            student_confidences = getattr(base_model, "_last_conf2_8x", None)
        else:
            student_features = None
            student_confidences = None

    # ========== PREPARE INPUT FOR SEMANTIC CONSISTENCY LOSS ==========
    if type(criterion).__name__ == "SemanticConsistencyLoss":
        def _select_per_view(feats, view_idx, batch, n_views):
            if feats is None:
                return None
            # List/Tuple: già per-view
            if isinstance(feats, (list, tuple)):
                return feats[view_idx]

            if torch.is_tensor(feats):
                # (n_views, B, C, H, W)
                if feats.dim() == 5:
                    return feats[view_idx]

                # (B, C, H, W) oppure (n_views*B, C, H, W)
                if feats.dim() == 4:
                    B = batch[view_idx]["img"].shape[0] if "img" in batch[view_idx] else None
                    if B is not None and feats.shape[0] == n_views * B:
                        feats = feats.view(n_views, B, *feats.shape[1:])
                        return feats[view_idx]
                    return feats

            # Fallback
            return feats

        # Special handling for semantic consistency loss
        for i in range(len(batch)):
            # Add teacher features to batch (BCHW format)
            if teacher_features is not None:
                batch[i]["semantics"] = _select_per_view(teacher_features, i, batch, len(batch))

            # Add student features to preds (BCHW format)
            if student_features is not None:
                preds[i]["semantics"] = _select_per_view(student_features, i, batch, len(batch))

            # [MODIFICA] Add student confidence to preds
            if student_confidences is not None:
                preds[i]["sem_conf"] = _select_per_view(student_confidences, i, batch, len(batch))

            # Fallback (Uniforme)
            else:
                B, C, H, W = preds[i]["semantics"].shape
                device = preds[i]["semantics"].device
                preds[i]["sem_conf"] = torch.ones(B, 1, H, W, device=device)
    # ==============================================================================
    if save_pca_visualization_path is not None:
        pca_visualization(batch, preds, epoch=epoch, output_dir=save_pca_visualization_path)
        pca_visualization_student_only(batch, preds, epoch, save_pca_visualization_path)
    
    with torch.autocast("cuda", enabled=False):
        loss = criterion(batch, preds) if criterion is not None else None

    result = {f"view{i + 1}": view for i, view in enumerate(batch)}
    result.update({f"pred{i + 1}": pred for i, pred in enumerate(preds)})
    result["loss"] = loss
    return result[ret] if ret else result

def validate_input_views_for_inference(
    views: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Strict validation and preprocessing of input views.

    Args:
        views: List of view dictionaries

    Returns:
        Validated and preprocessed views

    Raises:
        ValueError: For invalid keys, missing required keys, conflicting inputs, or invalid camera pose constraints
    """
    # Ensure input is not empty
    if not views:
        raise ValueError("At least one view must be provided")

    # Track which views have camera poses
    views_with_poses = []

    # Validate each view
    for view_idx, view in enumerate(views):
        # Check for invalid keys
        provided_keys = set(view.keys())
        invalid_keys = provided_keys - ALLOWED_VIEW_KEYS
        if invalid_keys:
            raise ValueError(
                f"View {view_idx} contains invalid keys: {invalid_keys}. "
                f"Allowed keys are: {sorted(ALLOWED_VIEW_KEYS)}"
            )

        # Check for missing required keys
        missing_keys = REQUIRED_KEYS - provided_keys
        if missing_keys:
            raise ValueError(f"View {view_idx} missing required keys: {missing_keys}")

        # Check for conflicting keys
        for conflict_set in CONFLICTING_KEYS:
            present_conflicts = [key for key in conflict_set if key in provided_keys]
            if len(present_conflicts) > 1:
                raise ValueError(
                    f"View {view_idx} contains conflicting keys: {present_conflicts}. "
                    f"Only one of {conflict_set} can be provided at a time."
                )

        # Check depth constraint: If depth is provided, intrinsics or ray_directions must also be provided
        if "depth_z" in provided_keys:
            if (
                "intrinsics" not in provided_keys
                and "ray_directions" not in provided_keys
            ):
                raise ValueError(
                    f"View {view_idx} depth constraint violation: If 'depth_z' is provided, "
                    f"then 'intrinsics' or 'ray_directions' must also be provided. "
                    f"Z Depth values require camera calibration information to be meaningful for an image."
                )

        # Track views with camera poses
        if "camera_poses" in provided_keys:
            views_with_poses.append(view_idx)

    # Cross-view constraint: If any view has camera_poses, view 0 must have them too
    if views_with_poses and 0 not in views_with_poses:
        raise ValueError(
            f"Camera pose constraint violation: Views {views_with_poses} have camera_poses, "
            f"but view 0 (reference view) does not. When using camera_poses, the first view "
            f"must also provide camera_poses to serve as the reference frame."
        )

    return views

def preprocess_input_views_for_inference(
    views: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Pre-process input views to match the expected internal input format.

    The following steps are performed:
    1. Convert intrinsics to ray directions when required. If ray directions are already provided, unit normalize them.
    2. Convert depth_z to depth_along_ray
    3. Convert camera_poses to the expected input keys (camera_pose_quats and camera_pose_trans)
    4. Default is_metric_scale to True when not provided

    Args:
        views: List of view dictionaries

    Returns:
        Preprocessed views with consistent internal format
    """
    processed_views = []

    for view_idx, view in enumerate(views):
        # Copy the view dictionary to avoid modifying the original input
        processed_view = dict(view)

        # Step 1: Convert intrinsics to ray_directions when required. If ray directions are provided, unit normalize them.
        if "intrinsics" in view:
            images = view["img"]
            height, width = images.shape[-2:]
            intrinsics = view["intrinsics"]
            _, ray_directions = get_rays_in_camera_frame(
                intrinsics=intrinsics,
                height=height,
                width=width,
                normalize_to_unit_sphere=True,
            )
            processed_view["ray_directions"] = ray_directions
            del processed_view["intrinsics"]
        elif "ray_directions" in view:
            ray_directions = view["ray_directions"]
            ray_norm = torch.norm(ray_directions, dim=-1, keepdim=True)
            processed_view["ray_directions"] = ray_directions / (ray_norm + 1e-8)

        # Step 2: Convert depth_z to depth_along_ray
        if "depth_z" in view:
            depth_z = view["depth_z"]
            ray_directions = processed_view["ray_directions"]
            ray_directions_unit_plane = ray_directions / ray_directions[..., 2:3]
            pts3d_cam = depth_z[..., None] * ray_directions_unit_plane
            depth_along_ray = torch.norm(pts3d_cam, dim=-1, keepdim=True)
            processed_view["depth_along_ray"] = depth_along_ray
            del processed_view["depth_z"]

        # Step 3: Convert camera_poses to expected input keys
        if "camera_poses" in view:
            camera_poses = view["camera_poses"]
            if isinstance(camera_poses, tuple) and len(camera_poses) == 2:
                quats, trans = camera_poses
                processed_view["camera_pose_quats"] = quats
                processed_view["camera_pose_trans"] = trans
            elif torch.is_tensor(camera_poses) and camera_poses.shape[-2:] == (4, 4):
                rotation_matrices = camera_poses[:, :3, :3]
                translation_vectors = camera_poses[:, :3, 3]
                quats = rotation_matrix_to_quaternion(rotation_matrices)
                processed_view["camera_pose_quats"] = quats
                processed_view["camera_pose_trans"] = translation_vectors
            else:
                raise ValueError(
                    f"View {view_idx}: camera_poses must be either a tuple of (quats, trans) "
                    f"or a tensor of (B, 4, 4) transformation matrices."
                )
            del processed_view["camera_poses"]

        # Step 4: Default is_metric_scale to True when not provided
        if "is_metric_scale" not in processed_view:
            # Get batch size from the image tensor
            batch_size = view["img"].shape[0]
            # Default to True for all samples in the batch
            processed_view["is_metric_scale"] = torch.ones(
                batch_size, dtype=torch.bool, device=view["img"].device
            )

        # Rename keys to match expected model input format
        if "ray_directions" in processed_view:
            processed_view["ray_directions_cam"] = processed_view["ray_directions"]
            del processed_view["ray_directions"]

        # Append the processed view to the list
        processed_views.append(processed_view)

    return processed_views

def postprocess_model_outputs_for_inference(
    raw_outputs: List[Dict[str, torch.Tensor]],
    input_views: List[Dict[str, Any]],
    apply_mask: bool = True,
    mask_edges: bool = True,
    edge_normal_threshold: float = 5.0,
    edge_depth_threshold: float = 0.03,
    apply_confidence_mask: bool = False,
    confidence_percentile: float = 10,
) -> List[Dict[str, torch.Tensor]]:
    """
    Post-process raw model outputs by copying raw outputs and adding essential derived fields.

    This function simplifies the raw model outputs by:
    1. Copying all raw outputs as-is
    2. Adding denormalized images (img_no_norm)
    3. Adding Z depth (depth_z) from camera frame points
    4. Recovering pinhole camera intrinsics from ray directions
    5. Adding camera pose matrices (camera_poses) if pose data is available
    6. Applying mask to dense geometry outputs if requested (supports edge masking and confidence masking)

    Args:
        raw_outputs: List of raw model output dictionaries, one per view
        input_views: List of original input view dictionaries, one per view
        apply_mask: Whether to apply non-ambiguous mask to dense outputs. Defaults to True.
        mask_edges: Whether to compute an edge mask based on normals and depth and apply it to the output. Defaults to True.
        apply_confidence_mask: Whether to apply the confidence mask to the output. Defaults to False.
        confidence_percentile: The percentile to use for the confidence threshold. Defaults to 10.

    Returns:
        List of processed output dictionaries containing:
            - All original raw outputs (after masking dense geometry outputs if requested)
            - 'img_no_norm': Denormalized RGB images (B, H, W, 3)
            - 'depth_z': Z depth from camera frame (B, H, W, 1) if points in camera frame available
            - 'intrinsics': Recovered pinhole camera intrinsics (B, 3, 3) if ray directions available
            - 'camera_poses': 4x4 pose matrices (B, 4, 4) if pose data available
            - 'mask': comprehensive mask for dense geometry outputs (B, H, W, 1) if requested
            - 'semantic_features': raw semantic features (B, H, W, 256) if feat_8x available
            - 'semantic_pca_rgb': PCA-reduced RGB visualization (B, H, W, 3) if feat_8x available
            - 'semantic_clusters': HDBSCAN cluster labels (B, H, W) if feat_8x available

    """
    processed_outputs = []

    for view_idx, (raw_output, original_view) in enumerate(
        zip(raw_outputs, input_views)
    ):
        # Start by copying all raw outputs
        processed_output = dict(raw_output)

        # 1. Add denormalized images
        img = original_view["img"]  # Shape: (B, 3, H, W)
        data_norm_type = original_view["data_norm_type"][0]
        img_hwc = rgb(img, data_norm_type)

        # Convert numpy back to torch if needed (rgb returns numpy)
        if isinstance(img_hwc, np.ndarray):
            img_hwc = torch.from_numpy(img_hwc).to(img.device)

        processed_output["img_no_norm"] = img_hwc

        # 2. Add Z depth if we have camera frame points
        if "pts3d_cam" in processed_output:
            processed_output["depth_z"] = processed_output["pts3d_cam"][..., 2:3]

        # 3. Recover pinhole camera intrinsics from ray directions if available
        if "ray_directions" in processed_output:
            intrinsics = recover_pinhole_intrinsics_from_ray_directions(
                processed_output["ray_directions"]
            )
            processed_output["intrinsics"] = intrinsics

        # 4. Add camera pose matrices if both translation and quaternions are available
        if "cam_trans" in processed_output and "cam_quats" in processed_output:
            cam_trans = processed_output["cam_trans"]  # (B, 3)
            cam_quats = processed_output["cam_quats"]  # (B, 4)
            batch_size = cam_trans.shape[0]

            # Convert quaternions to rotation matrices
            rotation_matrices = quaternion_to_rotation_matrix(cam_quats)  # (B, 3, 3)

            # Create 4x4 pose matrices
            pose_matrices = (
                torch.eye(4, device=img.device).unsqueeze(0).repeat(batch_size, 1, 1)
            )
            pose_matrices[:, :3, :3] = rotation_matrices
            pose_matrices[:, :3, 3] = cam_trans

            processed_output["camera_poses"] = pose_matrices  # (B, 4, 4)

        # ========== PROCESS SEMANTIC FEATURES (feat_8x) ==========
        # Process semantic features if available (from distilled SAM2 DPT head)
        if "feat_8x" in processed_output and SKLEARN_AVAILABLE:
            feat_8x = processed_output["feat_8x"]  # (B, 256, 64, 64)
            B, C, H_feat, W_feat = feat_8x.shape
            H_img, W_img = img.shape[-2:]
            
            # Upsample to image resolution if necessary
            if (H_feat, W_feat) != (H_img, W_img):
                feat_8x_upsampled = torch.nn.functional.interpolate(
                    feat_8x,
                    size=(H_img, W_img),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                feat_8x_upsampled = feat_8x
            
            # Convert to HWC format for alignment with pts3d
            semantic_features = feat_8x_upsampled.permute(0, 2, 3, 1).contiguous()  # (B, H, W, 256)

            # ===== COMPUTE COMMON PCA BASIS FROM ALL BATCH ELEMENTS =====
            all_feats_list = []
            for b_idx in range(B):
                feat_b = semantic_features[b_idx].cpu().numpy()  # (H, W, 256)
                H, W, feat_dim = feat_b.shape
                feat_flat = feat_b.reshape(-1, feat_dim).astype(np.float32, copy=False)
                all_feats_list.append(feat_flat)

            all_feats_combined = np.concatenate(all_feats_list, axis=0)  # (H*W*B, 256)

            # Compute common PCA basis and global normalization bounds
            common_pca_basis = None
            try:
                pca_common = PCA(n_components=3)
                pca_common.fit(all_feats_combined)
                
                # Project all features to compute global min/max
                all_proj_list = []
                for b_idx in range(B):
                    feat_b = semantic_features[b_idx].cpu().numpy()
                    H, W, feat_dim = feat_b.shape
                    feat_flat = feat_b.reshape(-1, feat_dim).astype(np.float32, copy=False)
                    feat_centered = feat_flat - pca_common.mean_
                    feat_pca = feat_centered @ pca_common.components_.T
                    all_proj_list.append(feat_pca)
                
                all_proj_combined = np.concatenate(all_proj_list, axis=0)
                global_min = all_proj_combined.min(axis=0, keepdims=True)
                global_max = all_proj_combined.max(axis=0, keepdims=True)
                
                common_pca_basis = {
                    "mean": pca_common.mean_,
                    "components": pca_common.components_.T,
                    "global_min": global_min,
                    "global_max": global_max,
                }
            except Exception as e:
                warnings.warn(f"Common PCA fitting failed: {e}")
            
            # Store raw semantic features
            processed_output["semantic_features"] = semantic_features
            
            # Process each batch element independently
            semantic_pca_rgb_list = []
            semantic_clusters_list = []
            
            for b in range(B):
                feat_b = semantic_features[b].cpu().numpy()  # (H, W, 256)
                H, W, C = feat_b.shape
                
                # Reshape to (N, 256) for processing
                feat_flat = feat_b.reshape(-1, C)  # (H*W, 256)
                
                # ===== GLOBAL PCA PROJECTION: 256D → 3D =====
                try:
                    feat_b = semantic_features[b].cpu().numpy()  # (H, W, 256)
                    H, W, C = feat_b.shape
                    feat_flat = feat_b.reshape(-1, C).astype(np.float32, copy=False)
                    
                    # Project using common basis
                    if common_pca_basis is not None:
                        feat_centered = feat_flat - common_pca_basis["mean"]
                        feat_pca = feat_centered @ common_pca_basis["components"]  # (H*W, 3)
                        
                        # Normalize to [0, 1] using global bounds
                        feat_pca_norm = (feat_pca - common_pca_basis["global_min"]) / (common_pca_basis["global_max"] - common_pca_basis["global_min"] + 1e-8)
                        pca_rgb = feat_pca_norm.reshape(H, W, 3)
                    else:
                        pca_rgb = np.zeros((H, W, 3))
                    
                    semantic_pca_rgb_list.append(torch.from_numpy(pca_rgb).float())
                except Exception as e:
                    warnings.warn(f"PCA projection failed for batch {b}: {e}")
                    semantic_pca_rgb_list.append(torch.zeros(H, W, 3))
                
                # ===== HDBSCAN: Cluster semantic features =====
                try:
                    total_points = H * W
                    stride = max(1, int(np.ceil(np.sqrt(total_points / MAX_HDBSCAN_POINTS))))
                    # Use only globally-aligned PCA representation as clustering input.
                    feat_small = pca_rgb[::stride, ::stride, :]
                    hs, ws, _ = feat_small.shape
                    feat_small_cluster = feat_small.reshape(-1, 3).astype(np.float32, copy=False)

                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=50,  # Adjust based on image size
                        min_samples=10,
                        metric='euclidean',
                        cluster_selection_epsilon=0.0,
                    )
                    cluster_labels_small = clusterer.fit_predict(feat_small_cluster)

                    clusters_small = cluster_labels_small.reshape(hs, ws)
                    clusters = np.repeat(
                        np.repeat(clusters_small, stride, axis=0),
                        stride,
                        axis=1,
                    )[:H, :W]
                    semantic_clusters_list.append(torch.from_numpy(clusters).long())
                except Exception as e:
                    warnings.warn(f"HDBSCAN failed for batch {b}: {e}")
                    # Fallback: all noise (-1)
                    semantic_clusters_list.append(torch.full((H, W), -1, dtype=torch.long))
            
            # Stack batch results
            semantic_pca_rgb = torch.stack(semantic_pca_rgb_list, dim=0).to(feat_8x.device)  # (B, H, W, 3)
            semantic_clusters = torch.stack(semantic_clusters_list, dim=0).to(feat_8x.device)  # (B, H, W)
            
            processed_output["semantic_pca_rgb"] = semantic_pca_rgb
            processed_output["semantic_clusters"] = semantic_clusters

        # 5. Apply comprehensive mask to dense geometry outputs if requested
        if apply_mask:
            final_mask = None

            # Start with non-ambiguous mask if available
            if "non_ambiguous_mask" in processed_output:
                non_ambiguous_mask = (
                    processed_output["non_ambiguous_mask"].cpu().numpy()
                )  # (B, H, W)
                final_mask = non_ambiguous_mask

            # Apply confidence mask if requested and available
            if apply_confidence_mask and "conf" in processed_output:
                confidences = processed_output["conf"].cpu()  # (B, H, W)
                # Compute percentile threshold for each batch element
                batch_size = confidences.shape[0]
                conf_mask = torch.zeros_like(confidences, dtype=torch.bool)
                percentile_threshold = (
                    torch.quantile(
                        confidences.reshape(batch_size, -1),
                        confidence_percentile / 100.0,
                        dim=1,
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )  # Shape: (B, 1, 1)

                # Compute mask for each batch element
                conf_mask = confidences > percentile_threshold
                conf_mask = conf_mask.numpy()

                if final_mask is not None:
                    final_mask = final_mask & conf_mask
                else:
                    final_mask = conf_mask

            # Apply edge mask if requested and we have the required data
            if mask_edges and final_mask is not None and "pts3d" in processed_output:
                # Get 3D points for edge computation
                pred_pts3d = processed_output["pts3d"].cpu().numpy()  # (B, H, W, 3)
                batch_size, height, width = final_mask.shape

                edge_masks = []
                for b in range(batch_size):
                    batch_final_mask = final_mask[b]  # (H, W)
                    batch_pts3d = pred_pts3d[b]  # (H, W, 3)

                    if batch_final_mask.any():  # Only compute if we have valid points
                        # Compute normals and normal-based edge mask
                        normals, normals_mask = points_to_normals(
                            batch_pts3d, mask=batch_final_mask
                        )
                        normal_edges = normals_edge(
                            normals, tol=edge_normal_threshold, mask=normals_mask
                        )

                        # Compute depth-based edge mask
                        depth_z = (
                            processed_output["depth_z"][b].squeeze(-1).cpu().numpy()
                        )
                        depth_edges = depth_edge(
                            depth_z, rtol=edge_depth_threshold, mask=batch_final_mask
                        )

                        # Combine both edge types
                        edge_mask = ~(depth_edges & normal_edges)
                        edge_masks.append(edge_mask)
                    else:
                        # No valid points, keep all as invalid
                        edge_masks.append(np.zeros_like(batch_final_mask, dtype=bool))

                # Stack batch edge masks and combine with final mask
                edge_mask = np.stack(edge_masks, axis=0)  # (B, H, W)
                final_mask = final_mask & edge_mask

            # Apply final mask to dense geometry outputs if we have a mask
            if final_mask is not None:
                # Convert mask to torch tensor
                final_mask_torch = torch.from_numpy(final_mask).to(
                    processed_output["pts3d"].device
                )
                final_mask_torch = final_mask_torch.unsqueeze(-1)  # (B, H, W, 1)

                # Apply mask to dense geometry outputs (zero out invalid regions)
                dense_geometry_keys = [
                    "pts3d",
                    "pts3d_cam",
                    "depth_along_ray",
                    "depth_z",
                ]
                for key in dense_geometry_keys:
                    if key in processed_output:
                        processed_output[key] = processed_output[key] * final_mask_torch

                # Apply mask to semantic features (zero out invalid regions)
                semantic_keys_3d = ["semantic_features", "semantic_pca_rgb"]
                for key in semantic_keys_3d:
                    if key in processed_output:
                        processed_output[key] = processed_output[key] * final_mask_torch
                
                # Apply mask to semantic clusters (set invalid to -1 = noise)
                if "semantic_clusters" in processed_output:
                    final_mask_2d = final_mask_torch.squeeze(-1)  # (B, H, W)
                    processed_output["semantic_clusters"] = torch.where(
                        final_mask_2d,
                        processed_output["semantic_clusters"],
                        torch.full_like(processed_output["semantic_clusters"], -1)
                    )

                # Add mask to processed output
                processed_output["mask"] = final_mask_torch

        processed_outputs.append(processed_output)

    return processed_outputs


def add_global_semantic_pca_rgb(
    processed_outputs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Compute global semantic PCA visualization across all views and add to outputs.
    
    Aligns with distillation visualization: computes a common PCA basis from all
    feature maps (feat_8x, 64x64x256) concatenated, then projects and normalizes
    each view using global bounds.
    
    Args:
        processed_outputs: List of output dicts from postprocess_model_outputs_for_inference
    
    Returns:
        List of output dicts with added 'semantic_pca_rgb_global' field (H, W, 3) [0, 1]
    """
    # === STEP 1: Collect all feat_8x from all views ===
    all_feats_list = []
    all_shapes = []
    
    for view_idx, output in enumerate(processed_outputs):
        if "feat_8x" not in output:
            continue
        
        feat_8x = output["feat_8x"]  # (B, C, H, W), typically (1, 256, 64, 64)
        if feat_8x.dim() == 4:
            feat_8x = feat_8x[0]  # Take first batch element: (C, 64, 64)
        
        # Convert to (H, W, C) then flatten to (H*W, C)
        feats = feat_8x.permute(1, 2, 0).contiguous()  # (H, W, 256)
        H, W, C = feats.shape
        all_shapes.append((H, W, C))
        feats_flat = feats.reshape(-1, C).float()  # (H*W, 256)
        all_feats_list.append(feats_flat)
    
    if not all_feats_list:
        # No feat_8x available, return unchanged
        return processed_outputs
    
    # === STEP 2: Compute common PCA basis from all concatenated features ===
    all_feats_combined = torch.cat(all_feats_list, dim=0).cpu()  # (H*W*N, 256)
    all_feats_combined = all_feats_combined.float()
    
    # PCA reduction: 256D -> 3D using torch.pca_lowrank
    U, S, V = torch.pca_lowrank(all_feats_combined, q=3, center=True)
    common_mean = all_feats_combined.mean(0)  # (256,)
    pca_basis = V[:, :3]  # (256, 3)
    
    # === STEP 3: Project all features and compute global bounds ===
    all_proj_list = []
    for feats_flat in all_feats_list:
        feats_flat = feats_flat.cpu().float()
        feats_centered = feats_flat - common_mean
        proj = feats_centered @ pca_basis  # (H*W, 3)
        all_proj_list.append(proj)
    
    all_proj_combined = torch.cat(all_proj_list, dim=0)  # (H*W*N, 3)
    global_min = all_proj_combined.min(dim=0)[0]  # (3,)
    global_max = all_proj_combined.max(dim=0)[0]  # (3,)
    eps = 1e-8
    global_range = global_max - global_min + eps
    
    # === STEP 4: Project each view and normalize with global bounds ===
    proj_idx = 0
    for view_idx, output in enumerate(processed_outputs):
        if "feat_8x" not in output:
            processed_outputs[view_idx]["semantic_pca_rgb_global"] = None
            continue
        
        feat_8x = output["feat_8x"]
        if feat_8x.dim() == 4:
            feat_8x = feat_8x[0]
        
        # Get original shape
        H, W, C = all_shapes[proj_idx]
        
        # Project
        feats = feat_8x.permute(1, 2, 0).contiguous().reshape(-1, C)
        feats = feats.cpu().float()
        feats_centered = feats - common_mean
        proj = feats_centered @ pca_basis  # (H*W, 3)
        
        # Normalize with global bounds
        proj_norm = (proj - global_min) / global_range
        proj_norm = torch.clamp(proj_norm, 0.0, 1.0)
        
        # Reshape to (H, W, 3) and move to device
        pca_rgb = proj_norm.reshape(H, W, 3).to(feat_8x.device)
        processed_outputs[view_idx]["semantic_pca_rgb_global"] = pca_rgb
        
        proj_idx += 1
    
    return processed_outputs


def compute_global_semantic_clusters(
    outputs: List[Dict[str, Any]],
    conf_percentile: Optional[float] = None,
    voxel_divisor: float = 200.0,
    min_voxel_size: float = 0.02,
) -> List:
    """
    Cluster semantic features across all views in one scene-level pass.
    
    Computes voxel-based clustering of semantic features in 3D space using
    HDBSCAN on globally-aligned PCA features (shared basis across views).
    
    Args:
        outputs: List of model predictions with semantic_pca_rgb_global (preferred),
            or semantic_pca_rgb as fallback, plus depth_z, intrinsics, camera_poses, mask
        conf_percentile: Confidence percentile threshold (0-100) for filtering points
        voxel_divisor: Divides scene extent to determine voxel size
        min_voxel_size: Minimum voxel size
    
    Returns:
        List of cluster label maps (H, W) for each view, or None if clustering fails
    """
    if not SKLEARN_AVAILABLE:
        warnings.warn("sklearn and/or hdbscan not available. Returning None for clusters.")
        return [None for _ in outputs]
    
    def _l2_normalize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """L2 normalization for feature vectors."""
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / (norms + eps)
    
    scene_samples = []

    for view_idx, pred in enumerate(outputs):
        required_keys = {"depth_z", "intrinsics", "camera_poses", "mask"}
        if not required_keys.issubset(pred.keys()):
            scene_samples.append(None)
            continue

        if "semantic_pca_rgb_global" in pred and pred["semantic_pca_rgb_global"] is not None:
            semantic_features = pred["semantic_pca_rgb_global"]
        elif "semantic_pca_rgb" in pred and pred["semantic_pca_rgb"] is not None:
            warnings.warn(
                "semantic_pca_rgb_global not found; falling back to semantic_pca_rgb for clustering.",
                RuntimeWarning,
            )
            semantic_features = pred["semantic_pca_rgb"]
        else:
            warnings.warn(
                f"Missing semantic_pca_rgb_global for view {view_idx}; skipping this view in global clustering.",
                RuntimeWarning,
            )
            scene_samples.append(None)
            continue

        depthmap_torch = pred["depth_z"][0].squeeze(-1)
        intrinsics_torch = pred["intrinsics"][0]
        camera_pose_torch = pred["camera_poses"][0]

        pts3d_world, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()

        if torch.is_tensor(semantic_features):
            semantic_features = semantic_features.detach().cpu().numpy()
        if semantic_features.ndim == 4:
            semantic_features = semantic_features[0]

        # Ensure semantic features match mask resolution.
        if semantic_features.shape[:2] != mask.shape:
            sem_tensor = torch.from_numpy(semantic_features).permute(2, 0, 1).unsqueeze(0).float()
            sem_resized = torch.nn.functional.interpolate(
                sem_tensor,
                size=mask.shape,
                mode="bilinear",
                align_corners=False,
            )
            semantic_features = sem_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()

        if conf_percentile is not None and "conf" in pred:
            conf_np = pred["conf"][0].squeeze().cpu().numpy()
            finite_conf = conf_np[np.isfinite(conf_np)]
            if finite_conf.size > 0:
                conf_threshold = np.nanpercentile(conf_np.reshape(-1), conf_percentile)
                mask = mask & np.isfinite(conf_np) & (conf_np >= conf_threshold)

        flat_mask = mask.reshape(-1)
        if not flat_mask.any():
            scene_samples.append(
                {
                    "shape": mask.shape,
                    "points": None,
                    "features": None,
                    "flat_mask": flat_mask,
                }
            )
            continue

        points = pts3d_world.cpu().numpy().reshape(-1, 3)
        features = semantic_features.reshape(-1, semantic_features.shape[-1])

        scene_samples.append(
            {
                "shape": mask.shape,
                "points": points[flat_mask],
                "features": features[flat_mask].astype(np.float32, copy=False),
                "flat_mask": flat_mask,
            }
        )

    valid_samples = [sample for sample in scene_samples if sample and sample["points"] is not None and len(sample["points"]) > 0]
    if not valid_samples:
        return [None for _ in outputs]

    points = np.concatenate([sample["points"] for sample in valid_samples], axis=0)
    features = np.concatenate([sample["features"] for sample in valid_samples], axis=0)

    scene_extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    voxel_size = max(scene_extent / voxel_divisor, min_voxel_size)
    origin = points.min(axis=0, keepdims=True)

    voxel_keys = np.floor((points - origin) / voxel_size).astype(np.int32)
    _, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    num_voxels = int(inverse.max()) + 1

    voxel_feature_sum = np.zeros((num_voxels, features.shape[1]), dtype=np.float32)
    voxel_counts = np.bincount(inverse, minlength=num_voxels).astype(np.float32)
    np.add.at(voxel_feature_sum, inverse, features)
    voxel_features = voxel_feature_sum / np.maximum(voxel_counts[:, None], 1.0)
    voxel_features = _l2_normalize(voxel_features)

    # No local PCA here: clustering runs directly on globally-aligned PCA features.
    cluster_input = voxel_features

    if cluster_input.shape[0] <= 1:
        voxel_labels = np.full(cluster_input.shape[0], -1, dtype=np.int32)
    else:
        min_cluster_size = max(8, min(40, cluster_input.shape[0] // 200))
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(1, min(min_cluster_size // 2, 5)),
            metric="euclidean",
            cluster_selection_method="leaf",
            cluster_selection_epsilon=0.0,
        )
        voxel_labels = clusterer.fit_predict(cluster_input).astype(np.int32, copy=False)

    sample_labels = voxel_labels[inverse]

    global_clusters_by_view = []
    cursor = 0
    for sample in scene_samples:
        if sample is None:
            global_clusters_by_view.append(None)
            continue

        h, w = sample["shape"]
        labels_view = np.full(h * w, -1, dtype=np.int32)

        if sample["points"] is not None and len(sample["points"]) > 0:
            count = len(sample["points"])
            labels_view[sample["flat_mask"]] = sample_labels[cursor : cursor + count]
            cursor += count

        global_clusters_by_view.append(labels_view.reshape(h, w))

    return global_clusters_by_view
