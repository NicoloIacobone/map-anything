# Plan: Increase Feature Map Resolution to Match Input Resolution

## Context

Currently, both teacher and student models produce fixed 64×64 feature maps, regardless of input image resolution. This creates a bottleneck for high-resolution feature extraction and semantic understanding. The goal is to achieve **full-resolution feature maps** (e.g., 224×224 input → 224×224 feature map) to preserve fine-grained semantic information.

**Why this matters:** Higher resolution feature maps enable better semantic segmentation, cleaner multi-view consistency, and improved distillation of fine details. This is especially critical for matching ground-truth camera poses and local geometry.

---

## Phase 1: SAM2 Native Higher-Resolution Features

### Investigation: Can SAM2 Natively Produce Higher Resolution Features?

**Current SAM2 Output:**
- **Input:** 1024×1024 images (configurable)
- **Hiera Backbone:** 4-stage vision transformer with progressive downsampling
  - Stage 0: stride 4 (256×256 for 1024×1024)
  - Stage 1: stride 8 (128×128 for 1024×1024)
  - Stage 2: stride 16 (64×64 for 1024×1024)
  - Stage 3: stride 32 (32×32 for 1024×1024)
- **FPN Neck:** Produces 4 feature pyramid levels (all 256-dimensional)
- **Current Usage:** Only uses `vision_features` (lowest resolution, stride 16)

**Higher Resolution Options:**

1. **Stride 4 Features (Native Maximum) ✓ RECOMMENDED**
   - Already available in `backbone_fpn[0]` 
   - For 1024×1024: 256×256 resolution
   - For 224×224: 56×56 resolution
   - High quality, no interpolation needed
   - Efficiently supported by SAM2 architecture
   
2. **Stride 2 Features:** Would require modifying Hiera backbone
   - Complex: Need to adjust `q_stride: (2, 2)` in config
   - Unclear efficiency implications
   - Lower priority alternative
   
3. **Stride 1 Features (Full Resolution):** **Not natively possible**
   - Would require removing all downsampling in Hiera
   - Would explode memory and computation
   - Not aligned with Hiera's hierarchical design

**Verdict:** Use **stride 4 features** as the native maximum from SAM2. These are already fully supported and provide excellent quality. For truly full-resolution output matching input dimensions, use post-processing interpolation.

---

## Phase 2: Interpolation Strategy for Full-Resolution Target

### Approaches to Reach Full-Resolution Output

**Option A: Interpolate SAM2 Stride-4 Features to Input Resolution**
- Extract stride 4 (naturally highest from SAM2)
- Bilinear interpolate to exact input dimensions
- Quality loss minimal (stride 4 already captures fine detail)
- Simple to implement
- Works at any resolution

**Option B: Modify DPT to Output Higher Resolution (Recommended)**
- Use stride 4 SAM2 features as teacher signal
- Modify DPT head output resolution from fixed 64×64 to stride 4 or higher
- Avoids aggressive interpolation of SAM2 features
- Better preserves teacher signal quality
- More complex implementation

**Recommendation:** Implement **Hybrid Approach**:
1. Extract stride 4 SAM2 features (native max quality)
2. Update DPT head to output at stride 4 resolution
3. Optionally interpolate both to exact input resolution for full-res output
4. This gives best quality without aggressive upsampling

---

## Phase 3: SAM2 Native Capability Analysis

### What Hiera Layers are Available?

From `/scratch2/nico/map-anything/sam2_minimal/sam2.1/sam2.1_hiera_l.yaml`:

```yaml
model:
  image_encoder:
    trunk:  # Hiera backbone
      stages: [2, 6, 36, 4]              # 4 stages with these block counts
      embed_dim: 144                      # Initial embedding dimension
      dim_mul: 2.0                        # Channel multiplier per stage
      head_mul: 2.0                       # Head multiplier per stage
      
    neck:   # Feature Pyramid Network
      d_model: 256                        # Output feature channels
      backbone_channel_list: [1152, 576, 288, 144]  # Channels per stage
      fpn_top_down_levels: [2, 3]         # Which levels get top-down fusion
      fpn_interp_model: nearest           # Upsampling method
```

**FPN Output Levels (all with 256 channels):**
- **Level 0 (stride 4):** 256×256 for 1024 input | 56×56 for 224 input | HIGHEST QUALITY ✓
- **Level 1 (stride 8):** 128×128 for 1024 input | 28×28 for 224 input
- **Level 2 (stride 16):** 64×64 for 1024 input | 16×16 for 224 input | CURRENT USAGE
- **Level 3 (stride 32):** 32×32 for 1024 input | 8×8 for 224 input

### How to Access Higher-Resolution Features

**Current Code (sam2_minimal/sam2_builder.py:220-236):**
```python
def forward(self, images):
    # ... preprocessing ...
    backbone_out = self.image_encoder(img_batch)
    return backbone_out["vision_features"]  # Returns level 3 (stride 32 lowest res)
```

**Change to stride 4:**
```python
# Currently backbone_out["vision_features"] is stride 16/32
# Need to access backbone_out["backbone_fpn"][0] for stride 4
```

**Available Features from backbone_out dictionary:**
- `"vision_features"`: Current stride-16 output
- `"backbone_fpn"`: List of 4 pyramid levels [stride-4, stride-8, stride-16, stride-32]
- `"vision_pos_enc"`: Position encodings for all levels

---

## Phase 4: Implementation Plan - Detailed Steps

### Part 1: Modify SAM2 Feature Extraction

**File:** `/scratch2/nico/map-anything/sam2_minimal/sam2_builder.py`

**Changes:**
1. **Update SAM2FeatureExtractor.forward()** (lines 220-236):
   - Add parameter: `use_high_res=True` (default)
   - If `use_high_res=True`: return stride-4 features `backbone_out["backbone_fpn"][0]`
   - If `use_high_res=False`: return stride-16 features for backward compatibility

2. **Update SAM2FeatureExtractor.__init__()** (lines 202-217):
   - Add parameter: `use_high_res=True`
   - Add parameter: `target_resolution=None` (auto-compute if None)

3. **Update load_sam2_feature_extractor()** (lines 239-273):
   - Pass `use_high_res` flag when creating SAM2FeatureExtractor
   - Document the resolution output

**Code locations:**
- `sam2_builder.py:202-236` ← Main changes
- `sam2_builder.py:239-273` ← Update function signature

### Part 2: Remove Hardcoded 64×64 in DPT Head

**File:** `/scratch2/nico/map-anything/mapanything/models/mapanything/model.py`

**Critical Issue:** Line 1397 hardcodes resolution
```python
target_hw_2 = (64, 64)  # ← THIS IS THE PROBLEM
```

**Changes:**
1. **Compute target resolution dynamically** (line 1397):
   ```python
   # Option A: Match input image dimensions
   input_h, input_w = images.shape[-2:]
   target_hw_2 = (input_h, input_w)
   
   # Option B: Use stride-4 resolution (recommended)
   stride = 4  # or from config
   target_hw_2 = (images.shape[-2] // stride, images.shape[-1] // stride)
   
   # Option C: Make configurable
   target_hw_2 = getattr(self.config, 'dpt_output_stride', 4)
   ```

2. **Update interpolation code** (lines 1418-1425):
   - Already handles variable sizes with bilinear interpolation
   - No changes needed if resolution is computed correctly

3. **Verify DPTFeature compatibility** (line 1407-1413):
   - Check if DPTFeature accepts variable `target_output_shape`
   - May need to verify upsampling logic

**Code locations:**
- `model.py:1397` ← Remove hardcoded value
- `model.py:1402-1425` ← Verify interpolation handles variable size

### Part 3: Update Loss Function

**File:** `/scratch2/nico/map-anything/train/losses.py`

**Considerations:**
1. Check SemanticConsistencyLoss handles variable resolutions
2. Verify multi-view correspondence matching (pts3d_cam) works at higher resolution
3. Search for hardcoded stride/resolution assumptions
4. Confidence weighting should work unchanged

**Key functions to verify:**
- `loss_of_one_batch_multi_view()`
- `SemanticConsistencyLoss.forward()`

**Likely changes needed:**
- May need to adjust loss weighting by resolution
- Verify 3D consistency matching adapts to feature grid size

### Part 4: Update Distillation Script

**File:** `/scratch2/nico/map-anything/scripts/distill.py`

**Changes:**
1. **Verify TeacherFeatureExtractor init** (lines 130-135):
   - Ensure `use_high_res` flag is passed if implementing Part 1
   
2. **Check feature shape handling** throughout:
   - Verify dataset loading respects variable resolution
   - Update any hardcoded 64×64 assumptions
   - Check logging/visualization code

3. **Add configuration option:**
   - Document output resolution in distillation config

### Part 5: Update Inference Script

**File:** `/scratch2/nico/map-anything/scripts/images_only_inference.py`

**Changes:**
1. **Feature upsampling** (check around lines 475-480):
   - Verify code adapts to variable feature resolution
   - May need to update target size computation
   
2. **PCA visualization** (lines 592-596):
   - Current code: reduces 256D → 3D for RGB
   - May be memory-intensive at higher resolution
   - Consider option to process patches

3. **HDBSCAN clustering:**
   - Verify clustering adapts to higher resolution
   - May need parameter tuning for higher density features

4. **Remove hardcoded 64×64:**
   - Search for any stride/resolution hardcoding

### Part 6: Configuration Updates

**Files to modify:**

1. **`/scratch2/nico/map-anything/configs/model/mapanything_distillation.yaml`:**
   - Add or verify `use_high_res_features: true`
   - Document feature output resolution

2. **`/scratch2/nico/map-anything/configs/model/pred_head/dpt_pose_scale_dpt2.yaml`:**
   - Add `output_stride: 4` or `output_resolution: null` (auto)

3. **`/scratch2/nico/map-anything/configs/distillation.yaml`:**
   - Document resolution settings

4. **`/scratch2/nico/map-anything/configs/loss/distillation_consistency.yaml`:**
   - Verify loss weighting works with new resolutions
   - May need resolution-dependent scaling

---

## Phase 5: Memory & Computation Impact

### Resolution Size Comparison

| Resolution | Memory vs Current | Compute vs Current | Notes |
|-----------|------------------|-------------------|-------|
| 64×64 (current) | 1× | 1× | Baseline |
| 112×112 (stride 4 @ 224) | 3× | 3× | Stride 4 for 224 input |
| 256×256 (stride 4 @ 1024) | 16× | 16× | Stride 4 for 1024 input |
| 224×224 (full res) | 12× | 12× | Full resolution |

### Mitigation Strategies
- Reduce batch sizes if memory-constrained
- Use gradient checkpointing in DPT head
- Process features in patches for visualization
- Consider mixed precision training

### Expected Quality Improvements
- Better semantic alignment (fewer quantization errors)
- Improved 3D consistency (higher geometric precision)
- Cleaner HDBSCAN clusters (higher feature density)

---

## Phase 6: Testing Strategy

### Unit Tests
1. SAM2FeatureExtractor with high-res flag
   - Verify stride-4 features have correct shape
   - Verify backward compatibility with high-res=False

2. DPT head with variable input
   - Test with 224×224, 448×448, 512×512, 1024×1024
   - Verify output shapes match configuration

3. SAM2CompatibilityLayer
   - Test with variable spatial dimensions
   - Verify confidence channel still produced correctly

4. Loss function
   - Test with different resolution features
   - Verify loss values are reasonable

### Integration Tests
1. Full distillation pipeline
   - Run training on test dataset
   - Check loss convergence
   - Verify no NaN/Inf values

2. Inference pipeline
   - Run images_only_inference with new resolution
   - Verify PCA and clustering work
   - Check output shapes

3. Multi-view consistency
   - Verify semantic features align across views
   - Check cluster stability across views

### Visual Quality Checks
1. PCA visualization comparison
   - Stride-4 vs stride-16 features
   - Verify semantic structure is preserved

2. Cluster visualization
   - Check HDBSCAN makes sense at higher resolution
   - Verify no spurious over-clustering

3. Geometry quality
   - Compare reconstructions at new resolution
   - Check pose accuracy

---

## Success Criteria

- ✅ SAM2 extracts stride-4 (or configurable) features
- ✅ Student DPT head produces matching resolution features
- ✅ No hardcoded (64, 64) in model.py
- ✅ Feature maps maintain correct dimensions through pipeline
- ✅ Loss computation works with variable resolution
- ✅ Distillation training runs without errors
- ✅ Inference produces semantically meaningful features
- ✅ No regression in existing functionality
- ✅ Configuration is clean and well-documented
- ✅ Memory usage is acceptable for target hardware

---

## Recommended Implementation Order

1. **Modify SAM2 feature extraction** (Part 1)
   - Low risk, isolated change
   - Unit test with stride-4 output

2. **Remove hardcoded 64×64 in DPT head** (Part 2)
   - Critical bottleneck
   - Test with multiple input sizes

3. **Verify loss function** (Part 3)
   - Run test batch through new pipeline
   - Check loss shapes and numerical stability

4. **Update distillation script** (Part 4)
   - Add configuration options
   - Update logging/visualization

5. **Update inference script** (Part 5)
   - Handle variable feature resolution
   - Test PCA and clustering

6. **Update configurations** (Part 6)
   - Document all changes
   - Make resolution configurable

7. **Integration testing** (Phase 6)
   - Full training pipeline
   - Inference quality assessment
   - Memory/compute profiling
