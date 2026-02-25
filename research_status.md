# Research Status: Towards Multi-View 3D Instance Segmentation via Semantic Distillation

## 1. Motivation & Problem Statement
* **The Gap:** 3D Instance Segmentation is currently hindered by the scarcity of densely annotated 3D datasets. Modern feed-forward approaches face a trade-off: they either lack metric geometric accuracy or fail to maintain high-resolution instance boundaries across multiple views without expensive test-time optimization.
* **Related Work 1 (UNITE):** A transformer-based framework that performs 3D scene understanding by distilling 2D foundation model features.
    * *Limitation:* Most implementations rely on CLIP-based distillation. While CLIP provides strong global semantic context (open-vocabulary), it inherently lacks the spatial precision required for accurate instance boundary delineation due to its image-level contrastive training objective.
* **Related Work 2 (PanSt3R):** A single-pass multi-view consistent panoptic segmentation model based on the DUSt3R lineage.
    * *Limitation:* As a specialized architecture trained on closed-set vocabularies, it may not fully exploit the "universal" segmentation knowledge embedded in recent large-scale foundation models like SAM 2.
* **Proposed Improvement (SAM 2 vs. CLIP):** Unlike CLIP, which optimizes for semantic similarity, **SAM 2** is trained on over 1 billion masks with an explicit pixel-level objective. By distilling SAM 2 features, our model inherits a significantly higher capacity for resolving fine-grained spatial details and instance separation, which is critical for downstream 3D clustering.

## 2. Methodology
### Architecture Overview
* **Student Model (MapAnything):** A VGGT-based architecture for metric 3D reconstruction.
    * **Encoder:** DINOv2 Large ($M$ blocks unfrozen).
    * **Info Sharing:** Multi-view Transformer ($N$ blocks unfrozen).
    * **Semantic DPT Head:** A Dense Prediction Transformer head added to regress:
        * **Student Features ($\mathbf{f}_s$):** Shape $[B, 256, 64, 64]$.
        * **Student Confidence ($\mathbf{c}_s$):** Shape $[B, 1, 64, 64]$ (Softplus activation).
* **Teacher Model (SAM 2):** Large version (Hiera + FPN Neck) used to generate target semantic/instance features $\mathbf{f}_t \in \mathbb{R}^{256 \times 64 \times 64}$.

### Mathematical Framework (Loss Functions)
1.  **Feature Distillation Loss ($L_{dist}$):** Minimizes the angular distance between student and teacher embedding vectors.
    $$L_{dist} = \frac{1}{|P|} \sum_{p \in P} \left( 1 - \frac{\mathbf{f}_{s,p} \cdot \mathbf{f}_{t,p}}{\|\mathbf{f}_{s,p}\| \|\mathbf{f}_{t,p}\|} \right)$$
    *Where $P$ is the set of pixels in the semantic map.*

2.  **Multi-View Consistency Loss ($L_{cons}$):** Enforces feature coherence across views via a geometry-aware Mean Consensus.
    * **Projection Mechanism:** Uses predicted local geometry ($\mathbf{pts}_{3D}^{cam}$) transformed by Ground Truth poses to find correspondences between View $i$ and View $j$.
    * **Target Computation (Consensus):**
        $$\bar{\mathbf{f}}_i = \frac{\sum_{j} \mathbf{v}_{j \to i} \cdot \mathbf{c}_{j \to i} \cdot \mathbf{f}_{j \to i}}{\sum_{j} \mathbf{v}_{j \to i} \cdot \mathbf{c}_{j \to i} + \epsilon}$$
    * **Loss Formulation:**
        $$L_{cons} = \frac{1}{|P_{valid}|} \sum_{p \in P_{valid}} \left( 1 - \text{cos\_sim}(\mathbf{f}_{i,p}, \text{sg}[\bar{\mathbf{f}}_{i,p}]) \right)$$
    * *Where $\mathbf{v}_{j \to i}$ is the validity mask (FOV check + Z-buffer occlusion check at 0.5m), $\mathbf{c}$ is the learned confidence, and $\text{sg}[\cdot]$ denotes the stop-gradient operator (as per UNITE methodology).*

## 3. Experiments & Implementation Details
* **Development Stages:**
    1.  **Single-View Validation:** Extensive testing on **COCO2017** (~118K images) to verify the DPT head's capacity to mimic SAM 2 features at scale (independent pre-validation).
    2.  **Multi-View Distillation (Current):** Conducted on a reduced **BlendedMVS** dataset (50 scenes for training, 5 for testing).
* **Freeze/Unfreeze Strategy ($M, N \in [0, 23]$):**
    * Ablation studies were performed by progressively unfreezing the last $M$ blocks of the DINOv2 Encoder and the last $N$ blocks of the Multi-View Transformer.
    * **Key Finding:** Increasing $M$ (Encoder blocks) has a drastically stronger impact on convergence speed and loss reduction compared to $N$ (Transformer blocks). This suggests that the geometric backbone's feature extraction capability must be adapted to support high-level semantic tasks.
* **Experimental Setup:**
    * **Hardware:** ETH Euler Cluster, 2x NVIDIA RTX 4090 GPUs.
    * **Precision:** Automatic Mixed Precision (AMP) using **bf16**.
    * **Optimizer:** AdamW ($\text{LR}_{head} = 5 \cdot 10^{-4}$, $\text{LR}_{encoder} = 5 \cdot 10^{-5}$).
* **Current Results:** Successfully achieved overfitting on single scenes (Loss $\approx 0$). Qualitative validation performed via **PCA visualization** of the distilled feature maps, showing spatially coherent coloring of instances across different views.

## 4. Conclusions & Next Steps
* **Insights:** The distillation from SAM 2 provides highly discriminative signals for instances, superior to global semantic embeddings. The dominance of the encoder unfreezing parameter ($M$) highlights the necessity of fine-tuning the foundational feature extractor.
* **Planned Developments:**
    * Full activation of **Consistency Loss** (`consistency_weight > 0`) to enforce global multi-view coherence.
    * Integration of **HDBSCAN** clustering on the 3D point cloud generated by MapAnything to produce discrete instance labels.
    * Development of a **Trainable Instance Head** to replace the manual clustering step, moving towards a fully end-to-end differentiable pipeline.

---

### Appendix: Scientific Motivation (SAM 2 vs. CLIP)
* **Boundary Awareness:** As demonstrated in *Kirillov et al., "Segment Anything" (2023)*, SAM is trained on over 1 billion masks with a specific IoU-based objective. This forces the model to encode precise spatial boundaries. In contrast, CLIP-based models (trained with contrastive loss) tend to produce "blob-like" activations that capture the general presence of an object but fail to delineate its exact physical extent (see *Li et al., "Grounding DINO", 2023*).
* **Feature Granularity:** SAM 2 utilizes a hierarchical Hiera encoder with a Feature Pyramid Network (FPN), preserving high-resolution spatial details essential for separating adjacent instances (e.g., books on a shelf). CLIP typically relies on standard ViT patch tokens ($16 \times 16$) which often merge features of small, neighboring objects.