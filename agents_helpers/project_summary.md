# MapAnything Semantic Distillation Project

## Overview
Extension of MapAnything multi-view 3D reconstruction with knowledge distillation from SAM2 encoder to enable semantic awareness. Student DPT head learns to predict semantic features aligned with teacher SAM2 encoder features through distillation and multi-view consistency losses.

---

## Architecture & Data Flow

### 1. MapAnything (Baseline)
Multi-view 3D reconstruction backbone generating geometric and semantic representations.

**Data Flow:**
- **Input:** N multi-view images (B, C=3, H_orig, W_orig) - commonly 224×224 or 448×448
- **Image Encoder (DINOv2):** RGB → hierarchical vision transformer features at multiple scales
  - Outputs: token embeddings with patch_size=14, so H_feat=H_orig/14, W_feat=W_orig/14
  - For 224×224 input: 16×16 feature tokens; for 448×448: 32×32
- **Geometric Encoders:** encode ray directions, depth, camera rotations/translations
- **Info Sharing (Multi-View Transformer):** cross-view attention fusion over N views
  - Input: (B×N_views, H_feat, W_feat, 1024) encoder features
  - Output: (B×N_views, H_feat, W_feat, 1024) fused multi-view features
- **DPT Head (Original):** Dense Prediction Transformer produces depth/normal/confidence
  - Projects features through multiple scales via upsampling layers
  - Output: (B, H_orig, W_orig, [depth, normals, confidence, mask])

**Geometric Features:** H_orig × W_orig resolution, highly coherent across views (MapAnything specialization)
**Semantic Features:** Originally none; geometric head does not capture object semantics

---

### 2. SAM2 Teacher Encoder
Frozen Segment Anything Model 2 encoder providing teacher semantic features.

**Data Flow:**
- **Input:** Same images as MapAnything, internally resized to 1024×1024
- **Backbone (Hiera):** Multi-scale vision transformer with FPN producing 4 stride levels
- **Output:** (B, 256, H_feat, W_feat) at stride-4 resolution
  - Features are 256-dim semantic embeddings
  - For 1024×1024 input: 256×256 (stride-4)
  - For 224×224 input: 56×56 (stride-4)
  - Dynamic resolution: H_feat=input_h/4, W_feat=input_w/4

**Role:** Ground truth semantic features capturing object-level semantics (learned from billions of mask annotations)

---

### 3. Student DPT Head + SAM2CompatibilityLayer (NEW)

**Architecture Components:**

#### dpt_feature_head_2 (Student DPT)
Second DPT head added to MapAnything for semantic predictions, parallel to geometric head.
- **Input:** (B, 1024, H_feat, W_feat) from multi-view transformer
  - H_feat×W_feat = (224/14)² = 16² or (448/14)² = 32² depending on resolution
- **DPT Processing:** Multi-scale feature aggregation
  - Projects input through 4 Conv1×1 layers → [256, 512, 1024, 1024] channels
  - Upsampling path: 4×, 2×, identity, 2× downsampling
  - FusionBlocks: multi-scale feature fusion via residual connections
  - Output conv: (B, 256, H_feat, W_feat)
- **Output:** (B, 256, H_feat_stride4, W_feat_stride4) semantic feature maps at stride-4 resolution
  - Dynamically computed: H_feat_stride4 = input_h/4, W_feat_stride4 = input_w/4
  - For 224×224 input: 56×56
  - For 448×448 input: 112×112
  - For 1024×1024 input: 256×256

**Resolution Behavior:**
- Training: features stay at stride-4 resolution (downsampled from input H_orig×W_orig via 4× stride)
- Inference: features upsampled to match input resolution using bilinear interpolation

#### SAM2CompatibilityLayer
Adapter matching student features to teacher feature format.
```
Input:  (B, 256, H_feat, W_feat) from DPT head
↓ LayerNorm(256, eps=1e-6)  → normalize per-channel across spatial dims
↓ Conv2d(256→256, kernel=1)  → project to SAM2 output space
Output: (B, 256, H_feat, W_feat) compatible with teacher features
```
- **Purpose:** Align feature distributions between student and teacher
- **Parameters:** minimal (1 LayerNorm + 1×1 Conv ≈ 256 + 256 params)

---

## Loss Functions

### SemanticConsistencyLoss (UNITE-style)
Combines two complementary objectives for semantic distillation.

#### 1. Distillation Loss (Eq. 2)
Pixel-level MSE between student and teacher semantic features.
```
L_distill = ||SAM2CompatibilityLayer(DPT_output) - SAM2_teacher||²
```
- **Resolution:** Computed at stride-4 resolution
  - For 224×224 input: 56×56
  - For 448×448 input: 112×112
  - For 1024×1024 input: 256×256
  - Student and teacher both produce stride-4 features (matched resolution)
- **Masking:** Optionally masked by confidence (min_conf threshold)
- **Weight:** `distillation_weight` (typically 1.0)
- **Effect:** Transfers semantic knowledge from 1B-param teacher to student

#### 2. Multi-View Consistency Loss (Eq. 4-5)
Enforces that features of the same 3D point are consistent across views.

**Algorithm:**
1. For anchor view i and target view j, compute 3D correspondences:
   - Project anchor pts3d (B, H_orig, W_orig, 3) into target view via camera poses
   - Grid sampling: match each pixel in anchor to its projection in target
   - Valid mask: in-frame, depth > 0, occlusion check via Z-buffer
2. Sample target features at projected locations: (B, 256, H_orig, W_orig)
   - Upsampled from stride-4 to full resolution
3. Compute weighted consistency loss:
   ```
   L_cons = ||f_anchor - CONF_weighted_mean(f_target)||²
   ```
   - Confidence-weighted averaging of multi-view features
   - Only applies to valid correspondences
- **Resolution:** Computed at H_orig × W_orig (image resolution)
  - Features upsampled from stride-4 to full resolution before consistency check
- **Weight:** `consistency_weight` (typically 0.1-1.0)
- **Effect:** Leverages geometric coherence (pts3d) to supervise semantic coherence

**Combined Loss:**
```
L_total = dist_weight × L_distill + cons_weight × L_cons
```

---

## Training Pipeline

### Data Flow (per batch):
1. **Load B images** at 224×224 (or 448×448)
2. **Student Forward Pass:**
   - DINOv2 encoder → (B, 1024, 16, 16) [or 32×32]
   - Multi-view transformer fusion → (B, 1024, 16, 16)
   - DPT head → (B, 256, stride-4 height, stride-4 width)
     - For 224×224: (B, 256, 56, 56)
     - For 448×448: (B, 256, 112, 112)
   - SAM2CompatibilityLayer → (B, 256, stride-4 h/w) student features
3. **Teacher Forward Pass (frozen):**
   - SAM2 encoder (1024×1024 internal) → (B, 256, stride-4 h/w) teacher features at stride-4
     - For any input: stride-4 resolution (256×256 for 1024 input, 56×56 for 224 input)
4. **Loss Computation:**
   - Distillation: student (stride-4) vs teacher (stride-4)
     - Both at same resolution (stride-4)
     - MSE at stride-4 resolution
   - Consistency: multi-view correspondence matching
     - Sample features across views at projected locations
     - Weighted mean aggregation via confidence
     - MSE at original resolution (224×224 or 448×448)
5. **Backward pass:** gradients only flow through:
   - dpt_feature_head_2
   - sam2_compat
   - Selected info_sharing transformer blocks (unfrozen)
   - Selected DINOv2 encoder blocks (unfrozen)

### Checkpoint Structure
Each epoch saves encoder-specific checkpoint:
```python
checkpoint = {
    "dpt_feature_head_2": student_head_state_dict,
    "sam2_compat": compatibility_layer_state_dict,
    "info_sharing_blocks": {idx: block_state_dict, ...},
    "dino_encoder_blocks": {idx: block_state_dict, ...},
    "epoch": epoch_number,
    "best_val_loss": best_loss_value,
}
```

---

## Inference Pipeline

### Feature Generation (demo_images_only_inference.py):
1. **Load images** → preprocess to 448×448 (norm: DINOv2 stats)
2. **Model inference:**
   - Forward through MapAnything with loaded encoder checkpoint
   - Returns semantic features at stride-4 resolution (112×112 for 448×448 input)
3. **Global PCA Basis:**
   - Concatenate features across all views
   - Fit PCA on full scene (global basis, not per-view)
   - Project to RGB for visualization
4. **Semantic Clustering (HDBSCAN):**
   - Cluster global semantic features
   - Each cluster = distinct semantic object
5. **SAM2 Mask Integration (optional):**
   - Run SAM2 video predictor on preprocessed images
   - Produce coherent masks across temporal sequence
   - Reproject 2D masks onto 3D point clouds
6. **Visualization (Rerun):**
   - 3D point clouds with RGB
   - Semantic PCA RGB overlays
   - HDBSCAN cluster colorization
   - SAM2 mask reprojections

---

## Key Implementation Details

### Feature Upsampling
```
Training/Validation:
- Semantic features generated at stride-4 (H_feat × W_feat where stride=4)
- DPT dynamically computes: H_feat = input_h/4, W_feat = input_w/4
- Consistency loss requires full resolution: upsample to H_orig × W_orig
- Upsampling mode: bilinear, align_corners=False

Inference:
- Features extracted at stride-4 resolution
- Upsampled to input image dimensions for full-resolution output
- Globally PCA-reduced for visualization
- HDBSCAN applied in feature space directly
```

### Multi-View Correspondence
```
For each pixel (u, v) in anchor view:
1. Get 3D point from pts3d_anchor (geometric from student)
2. Transform via GT camera poses to target view
3. Project via target intrinsics → (u_proj, v_proj)
4. Check validity: in-image, depth > 0, not occluded
5. Bilinear sample target semantic features at (u_proj, v_proj)
6. Compare via L2 loss
```

### Freeze Strategy
- **Frozen:** DINOv2 first N layers, most info_sharing blocks, geometric heads
- **Unfrozen:** Selected DINOv2 blocks, selected info_sharing blocks, dpt_feature_head_2, sam2_compat
- **Teacher:** SAM2 encoder always frozen during distillation

---

## Feature Dimensions Summary

| Component | Input Shape | Output Shape | Resolution |
|-----------|-------------|--------------|------------|
| Image Input | (B, 3, H_orig, W_orig) | - | 224/448 |
| DINOv2 Encoder | (B, 3, H_orig, W_orig) | (B, 1024, H_feat, W_feat) | 16/32 |
| Multi-view Transformer | (B, 1024, H_feat, W_feat) | (B, 1024, H_feat, W_feat) | 16/32 |
| DPT Head | (B, 1024, H_feat, W_feat) | (B, 256, H_orig/4, W_orig/4) | stride-4 |
| SAM2 Compat Layer | (B, 256, H_orig/4, W_orig/4) | (B, 256, H_orig/4, W_orig/4) | stride-4 |
| SAM2 Teacher | (B, 3, 1024, 1024) | (B, 256, 256, 256) | stride-4 (256×256) |
| Student Features (stride-4) | - | (B, 256, H_orig/4, W_orig/4) | stride-4 dynamic |
| Distillation Loss | L2(student_stride4, teacher_stride4) | scalar | stride-4 |
| Consistency Loss | Projected features + L2 | scalar | H_orig×W_orig |

**Resolution Examples:**
- 224×224 input → 56×56 stride-4 features
- 448×448 input → 112×112 stride-4 features
- 1024×1024 input → 256×256 stride-4 features

---

## Current Status

**Implemented:**
- ✅ DPT semantic head (dpt_feature_head_2) integrated into MapAnything
- ✅ SAM2CompatibilityLayer for feature alignment
- ✅ SemanticConsistencyLoss (distillation + multi-view consistency)
- ✅ Checkpoint save/load for encoder components
- ✅ SAM2 mask video prediction and reprojection on 3D scenes
- ✅ Global semantic PCA visualization
- ✅ HDBSCAN clustering on semantic features
- ✅ High-resolution feature map support (distillation_higher_resolution_518 test case)

**Latest Updates (Commit: 8d0c78a):**
- **Higher Resolution Feature Maps:** Increased resolution of feature map test case via distillation_higher_resolution_518 test configuration
- **Configuration Updates:** Modified configs/distillation.yaml and dataset/bmvs_reduced.yaml for improved resolution handling
- **Model Refinements:** Updated mapanything/models/mapanything/model.py to support variable resolution features
- **Loss Function Updates:** Enhanced mapanything/train/losses.py for better multi-scale feature handling
- **SAM2 Builder Improvements:** Refined sam2_minimal/sam2_builder.py with 50+ lines of optimization
- **Documentation:** Generated comprehensive IMPLEMENTATION_SUMMARY.md and new_features_size.md documenting feature dimensions and architecture
- **Script Organization:** Moved demo scripts to dedicated scripts/demos/ directory for better project structure
- **Training Infrastructure:** Added slurm/distillation_higher_resolution_518.sh for high-resolution feature map training runs

**Open Research Questions:**
- What is the optimal resolution for semantic feature maps (impact on accuracy vs. computation)?
- Decoder design for dense object segmentation (currently feature-only)
