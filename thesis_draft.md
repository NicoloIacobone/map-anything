## Abstract

This thesis addresses the fundamental gap between 3D reconstruction and high-level semantic understanding in computer vision. While modern feed-forward models like MapAnything excel at reconstructing dense geometry from multiple views, they lack the ability to interpret scene semantics. Conversely, 2D Foundation Models such as the Segment Anything Model 2 (SAM 2) provide unprecedented object segmentation capabilities but lack 3D coherence. To bridge this divide, this work investigates a semantic distillation framework. Drawing inspiration from state-of-the-art approaches like UNITE, we adapt a student-teacher architecture where a 3D-consistent geometric backbone serves as the student, tasked with regressing the high-dimensional latent space of a frozen SAM 2 teacher. The framework is designed to include both feature distillation and a multi-view consistency loss to enforce geometric coherence. Experimental validation focused on assessing the distillation capacity of the student model. We identify a critical "capacity bottleneck" in standard geometric encoders when tasked with fine-grained semantic regression. Our results demonstrate that adapting (unfreezing) the geometric backbone is strictly necessary to minimize the distillation loss and achieve high-fidelity semantic features. This analysis provides the foundational constraints and architectural insights required to build fully end-to-end 3D instance segmentation systems.

## 3. Methodology: Semantic Distillation Framework

This chapter provides a comprehensive technical description of the semantic distillation framework that combines 3D geometric understanding from MapAnything with high-level semantic understanding from SAM 2. We adopt a student-teacher architecture where a frozen 2D foundation model (teacher) guides the learning of a 3D geometric model (student), tasked with regressing high-dimensional semantic feature vectors in a geometrically consistent manner.

### 3.1 System Overview

**Conceptual Framework.** The semantic distillation framework addresses a fundamental architectural challenge: bridging the gap between geometric and semantic representations. The core insight is that while geometric models excel at dense 3D reconstruction from multiple views, they lack understanding of *what* is being reconstructed. Foundation models like SAM 2 provide rich semantic understanding of 2D objects and regions but fail to enforce 3D coherence. Our approach combines these capabilities through feature distillation, where the student model learns to predict dense semantic feature maps that align with the teacher while maintaining geometric consistency across views.

**High-Level Pipeline.** The system operates as follows:

1. **Multi-View Input**: The pipeline receives multiple views of a scene as RGB images.
2. **Teacher Feature Extraction**: For each 2D image, the frozen SAM 2 teacher model extracts dense feature maps of shape $(B, H', W', 256)$ at a downsampled resolution, which serves as the target for distillation.
3. **Geometric Feature Prediction**: The student model (MapAnything) processes multi-view images through its geometric encoder, aggregates information across views using a multi-view transformer, and produces dense semantic feature predictions that align with the teacher resolution.
4. **Dual Loss Computation**: Two loss objectives are optimized simultaneously:
   - **2D Distillation Loss**: Enforces per-pixel alignment between predicted and teacher features.
   - **Multi-View Consistency Loss**: Ensures that features of the same 3D point, when projected across different views, remain coherent and consistent.

The student model is encouraged to learn semantic features not through explicit supervision (which would require dense semantic annotations), but through implicit learning from a frozen teacher that outputs high-quality semantic features without task-specific fine-tuning.

---

### 3.2 The Teacher Model (SAM 2): Frozen Semantic Feature Extraction

**Architecture and Feature Extraction.** The teacher model is the SAM 2 image encoder, which consists of a Hiera vision backbone (hierarchical image encoder) followed by an FPN-style neck. This architecture is specifically designed for dense prediction tasks, outputting multi-scale feature maps. In our setup, we extract features at a specific intermediate layer of this backbone, chosen to balance semantic richness with computational tractability.

**Feature Dimensionality and Resolution.** The SAM 2 teacher provides feature tensors of shape:
$$\mathbf{F}_{\text{teacher}} \in \mathbb{R}^{B \times H_{\text{sam}} \times W_{\text{sam}} \times 256}$$

where $B$ is the batch size, $H_{\text{sam}}$ and $W_{\text{sam}}$ are the spatial dimensions (typically 64×64 for a 1024×1024 input image), and 256 is the fixed embedding dimension. This resolution represents a $16 \times$ downsampling from the original image size, a standard choice in vision transformers that balances computational cost with feature density.

**Frozen Weights.** The entire SAM 2 teacher is kept frozen throughout training. This design choice is motivated by several considerations:
- **Stability**: Keeping the teacher frozen ensures that the target distribution remains stable, preventing the student from chasing a moving target.
- **Computational Efficiency**: Avoiding backpropagation through the large teacher model reduces memory and computational overhead.
- **Self-Supervision Philosophy**: The teacher acts as a fixed, pre-trained feature extractor, a principle borrowed from self-supervised learning. The student learns to mimic this fixed reference without modifying it.

This is fundamentally different from standard supervised learning where we would fine-tune the teacher model. Here, we exploit the teacher's already-learned semantic knowledge and use it as a stable signal for the student.

---

### 3.3 The Student Model (Geometric Backbone): Architecture and Modifications

**Overall Architecture.** The student model is built upon MapAnything, a state-of-the-art multi-view 3D reconstruction framework. MapAnything consists of three main components:

1. **Image Encoder** (DINOv2 Large): Extracts geometric features from each input image independently.
2. **Multi-View Transformer** (Alternating Attention): Aggregates information across multiple views to build a globally consistent 3D scene representation.
3. **Prediction Heads** (DPT-based): Dense prediction heads that convert aggregated features into dense geometric predictions (depths, point maps, etc.).

In the standard MapAnything pipeline, these prediction heads produce geometric outputs (3D coordinates, depth, pose, etc.). For semantic distillation, we add a **parallel semantic prediction head** alongside the existing geometric heads.

**Feature Resolution Throughout the Pipeline.** Before describing the semantic head architecture, it is important to understand the feature resolution evolution. Consider an example with input images of size $518 \times 518$:

1. **DINOv2 Encoder**: Processes images by dividing them into non-overlapping patches of size $14 \times 14$. A $518 \times 518$ image is thus converted to a feature grid of size $37 \times 37$ (with $518 / 14 = 37$).
2. **Multi-View Transformer**: Aggregates features across multiple views and applies upsampling by a factor of 8. This transforms the $37 \times 37$ feature grid to resolution $296 \times 296$.
3. **Semantic Prediction Head**: Further processes these features and produces the final semantic predictions.

**Semantic Prediction Head Architecture.** To predict semantic features aligned with SAM 2, we introduce a two-stage process:

$$\text{SemanticHead} = \text{DPT}_2 \rightarrow \text{SAM2CompatibilityLayer}$$

**Stage 1: DPT Feature Head** ($\text{DPT}_2$). A dedicated DPT head processes the upsampled multi-view features (at $296 \times 296$ resolution) from the transformer. The DPT architecture is specifically designed for dense prediction tasks and provides several key advantages:

- **Multi-Scale Feature Fusion**: The DPT internally fuses features from multiple hierarchical levels of the transformer backbone, capturing both fine-grained details and global semantic context.
- **Spatial Regularity**: Unlike simple linear layers or point-wise convolutions, the DPT preserves spatial relationships and locality, which is crucial for maintaining geometric coherence in dense predictions.
- **Contextual Aggregation**: The architecture naturally aggregates contextual information across the spatial domain, which is important for producing semantically meaningful features.

The DPT head processes features of resolution $296 \times 296$ and maintains this resolution, outputting:
$$\mathbf{F}_{\text{dpt}} \in \mathbb{R}^{B \times D_{\text{dpt}} \times 296 \times 296}$$

where $D_{\text{dpt}} = 256$ is the internal feature dimension of the DPT head.

**Stage 1.5: Spatial Resolution Alignment (Manual Preprocessing).** A critical architectural step occurs before channel projection: the student produces features at resolution $296 \times 296$ (8× upsampling of the patch grid), while the teacher model operates at $64 \times 64$ resolution. To align spatial dimensions with the teacher, we apply bilinear interpolation as a preprocessing step:

$$\mathbf{F}_{\text{dpt}}^{\text{interp}} = \text{Interpolate}_{\text{bilinear}}\big(\mathbf{F}_{\text{dpt}}, \text{size}=(64, 64)\big)$$

This spatial downsampling is implemented as a **manual forward pass operation**, separate from and prior to the SAM2CompatibilityLayer module. Bilinear interpolation is chosen to smoothly aggregate spatial information while downsampling, preserving semantic coherence across the spatial domain.

**Stage 2: SAM2CompatibilityLayer (Channel Projection).** After spatial alignment, the SAM2CompatibilityLayer module handles the channel projection and feature normalization to align with SAM 2's learned feature space:

$$\text{SAM2CompatibilityLayer}(x) = \text{Conv}_{1 \times 1}(\text{LayerNorm}(\pi(x)))$$

This layer performs two sequential operations on the spatially-aligned features:

1. **Permutation and Normalization** ($\pi$): Reshapes feature maps from spatial format $(B, C, H, W)$ to sequence format, applies layer normalization across the channel dimension. This ensures numerical stability and consistency with SAM 2's learned feature distribution, which is implicitly normalized during SAM 2's training.

2. **Channel Projection** (Conv$_{1 \times 1}$): A 1×1 convolution projects the normalized features to exactly **257 dimensions**: 256 for semantic content and 1 for a per-pixel confidence score. The confidence channel allows the model to express uncertainty about which pixels have reliable semantic features, useful for weighting in the consistency loss.

The final output of the semantic head is now aligned both spatially and in channel dimension:
$$\mathbf{F}_{\text{pred}} \in \mathbb{R}^{B \times 64 \times 64 \times 257}$$

This output directly matches the teacher's spatial resolution ($64 \times 64$) while extending the channel dimension to include the confidence channel, enabling fair comparison during distillation loss computation versus $\mathbf{F}_{\text{teacher}} \in \mathbb{R}^{B \times 64 \times 64 \times 256}$.

**Why DPT?** We chose the DPT architecture for the semantic feature head over simpler alternatives (e.g., direct 1×1 convolutions or fully-connected layers) for several key reasons:

- **Geometric Inductive Bias**: DPT preserves spatial coherence and locality, which is essential for maintaining the geometric structure and consistency of the 3D scene representation.
- **Multi-Scale Hierarchical Processing**: By fusing features from multiple scales of the transformer backbone, DPT naturally captures both local texture details and global semantic context simultaneously.
- **Proven Performance**: DPT has demonstrated strong empirical performance in dense prediction tasks, including depth estimation, panoptic segmentation, and semantic segmentation.
- **Feature Expressiveness**: The multi-scale fusion and contextual aggregation in DPT produce more expressive and semantically rich feature representations compared to simpler convolutional alternatives.

---

### 3.4 Training Objective and Loss Functions

The student model is trained using a composite loss function that combines two complementary objectives: **2D feature distillation** and **multi-view geometric consistency**. Drawing inspiration from UNITE's semantic loss formulation, we adapt the framework to leverage ground-truth geometry for establishing 3D correspondences while maintaining semantic alignment with the frozen teacher model.

#### 3.4.1 Feature Distillation Loss

The distillation loss enforces that the student's predicted semantic features align with the teacher's frozen features at each spatial location. Given a view $i$ with predicted features $\mathbf{f}^i_s \in \mathbb{R}^{H \times W \times 256}$ (student) and teacher features $\mathbf{f}^i_t \in \mathbb{R}^{H \times W \times 256}$, we define the per-pixel distillation loss using cosine similarity:

$$\mathcal{L}_{\text{dist}}^i = \frac{1}{|\Omega|} \sum_{u \in \Omega} \left[ 1 - \cos(\mathbf{f}^i_s(u), \mathbf{f}^i_t(u)) \right]$$

where $\Omega$ represents the set of all spatial locations in the feature map, and $|\Omega|$ is its cardinality. The total distillation loss is computed as the average across all $N$ views:

$$\mathcal{L}_{\text{dist}} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_{\text{dist}}^i$$

**Rationale for Cosine Similarity.** Cosine similarity is preferred over $\ell_2$ distance for several reasons: (1) it measures angular alignment rather than magnitude, making it invariant to feature scale differences between teacher and student; (2) in high-dimensional embedding spaces (e.g., 256D), semantic similarity is effectively encoded directionally; (3) it provides stable gradients for optimization, avoiding issues with vanishing or exploding gradients common with MSE on normalized features.

#### 3.4.2 Multi-View Consistency Loss

Beyond 2D supervision, we enforce that semantic features of the same 3D point remain consistent across different viewpoints. This constraint ensures that the learned features are truly 3D-coherent rather than view-dependent artifacts.

**Correspondence Establishment via Ground-Truth Geometry.** For each anchor view $i$, we establish correspondences with other views using ground-truth 3D geometry. Given a pixel $u^i$ in view $i$ with corresponding world point $\mathbf{p}^i \in \mathbb{R}^3$, we compute its projection into view $j$ as:

$$u^j = \pi_j(\mathbf{p}^i) = \mathbf{K}_j \left[ \mathbf{R}_j \mid \mathbf{t}_j \right] \begin{bmatrix} \mathbf{p}^i \\ 1 \end{bmatrix}$$

where $\mathbf{K}_j$ is the intrinsic matrix, and $[\mathbf{R}_j \mid \mathbf{t}_j]$ is the world-to-camera extrinsic transformation for view $j$.

**Occlusion Handling.** To filter invalid correspondences due to occlusions, we perform a depth consistency check. Let $z^i_{\text{proj}}$ be the depth of point $\mathbf{p}^i$ when transformed into view $j$'s camera frame, and $z^j(u^j)$ be the ground-truth depth at the projected location $u^j$. A correspondence is considered valid if:

$$|\,z^i_{\text{proj}} - z^j(u^j)\,| < \tau_{\text{occ}}$$

where $\tau_{\text{occ}} = 0.5$ meters in our implementation. This threshold filters out points occluded by closer geometry.

**Consensus Feature Aggregation.** For each anchor pixel $u^i$ in view $i$, we aggregate features from all valid correspondences across views. Define the set of valid correspondences as:

$$\mathcal{C}(u^i) = \{(j, u^j) \mid j \neq i, \, u^j = \pi_j(\mathbf{p}^i), \, \text{valid}(u^i, j, u^j)\}$$

where $\text{valid}(\cdot)$ checks FOV boundaries and occlusion constraints. The consensus feature is computed as a confidence-weighted mean:

$$\bar{\mathbf{f}}(u^i) = \frac{\sum_{(j, u^j) \in \mathcal{C}(u^i)} c^j(u^j) \cdot \mathbf{f}^j_s(u^j) + c^i(u^i) \cdot \mathbf{f}^i_s(u^i)}{\sum_{(j, u^j) \in \mathcal{C}(u^i)} c^j(u^j) + c^i(u^i)}$$

where $c(u) \in [1, \infty)$ is the predicted per-pixel confidence (output from the 257th channel, activated via softplus+1 to ensure positivity and minimum base confidence). The consensus explicitly includes the anchor view itself, ensuring stable targets even with limited overlap.

**Consistency Loss Formulation.** Following UNITE's approach, we apply a stop-gradient operation to the consensus target $\bar{\mathbf{f}}(u^i)$ to prevent degenerate solutions where all features collapse to a constant. The consistency loss for view $i$ is evaluated only on valid overlapping pixels:

$$\mathcal{L}_{\text{cons}}^i = \frac{1}{|\mathcal{M}^i|} \sum_{u^i \in \mathcal{M}^i} \left[ 1 - \cos\left(\mathbf{f}^i_s(u^i), \text{sg}[\bar{\mathbf{f}}(u^i)]\right) \right]$$

where $\text{sg}[\cdot]$ denotes the stop-gradient operator, and $\mathcal{M}^i = \{u^i \mid |\mathcal{C}(u^i)| > 0\}$ is the subset of pixels with at least one valid multi-view correspondence. The total consistency loss is averaged across all views:

$$\mathcal{L}_{\text{cons}} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_{\text{cons}}^i$$

#### 3.4.3 Composite Loss Function

The total training objective combines both losses with their respective weighting coefficients:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{dist}} \cdot \mathcal{L}_{\text{dist}} + \lambda_{\text{cons}} \cdot \mathcal{L}_{\text{cons}}$$

where $\lambda_{\text{dist}} = \lambda_{\text{cons}}$ in our implementation, giving equal importance to 2D semantic alignment and 3D geometric consistency.

**Training Dynamics.** The distillation loss acts as a direct supervision signal, encouraging the student to mimic the teacher's semantic understanding. Concurrently, the consistency loss enforces that this understanding is geometrically coherent: features of the same 3D point must agree across views, weighted by their predicted confidence. The stop-gradient operation in the consistency target prevents feature collapse while allowing the model to learn robust, view-invariant representations.

**Difference from Standard Consistency Losses.** A key distinction of our framework is that we leverage **ground-truth 3D geometry and camera poses** to establish correspondences, rather than relying on predicted geometry or optical flow. This provides several critical advantages: (1) correspondences are geometrically accurate and independent of the student's current feature quality; (2) the consistency loss provides a stable training signal even in early training stages when the student's predictions might be unreliable; and (3) semantic features are forcibly aligned with the true 3D structure of the scene, preventing 2D view-dependent hallucinations from propagating across views.