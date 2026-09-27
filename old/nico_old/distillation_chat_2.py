"""
Distillation Training Script for MapAnything - SAM2 Encoder Knowledge Transfer

This script implements knowledge distillation from SAM2's encoder features into
an additional DPT head (dpt_feature_head_2) in the MapAnything model.

Architecture:
- Teacher: SAM2 encoder features (pre-computed and saved as .pt files)
- Student: dpt_feature_head_2 in MapAnything (feature_upsampled_8x output)
- Loss: Combination of MSE and Cosine Similarity on per-pixel features

References:
- MapAnything training: mapanything/train/training.py
- Original distillation script: distillation.py
"""
import argparse
import datetime
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torchvision import transforms

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Optional: psutil for CPU memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Optional: wandb for experiment tracking
try:
    import wandb  # type: ignore[import-not-found]
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore
    print("[WARN] wandb not installed. Logging to wandb disabled.")

from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils import train_tools
from nico.utils import mean_std_difference, create_student_original_teacher_side_by_side

from sam2_minimal.modeling.sam.mask_decoder import MaskDecoder
from sam2_minimal.modeling.sam.transformer import TwoWayTransformer
from sam2_minimal.modeling.sam.prompt_encoder import PromptEncoder

# Enable TF32 precision if supported
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True

def setup_runtime_paths(args):
    """Inizializza OUT_DIR, BASE_DIR, DATASET, SAM2_PATH e le directory immagini/feature usando args."""
    import torch.hub as _torch_hub
    global OUT_DIR, BASE_DIR, DATASET, SAM2_PATH, CONFIG_JSON_PATH
    global TRAIN_SPLIT, VAL_SPLIT
    global IMAGES_DIRNAME, FEATURES_DIRNAME
    global TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_FEATURES_DIR, VAL_FEATURES_DIR
    global run_cluster

    run_cluster = not sys.stdout.isatty()
    # STABLE_REV = getattr(args, "model_revision", "6f3a25bfbb8fcc799176bb01e9d07dfb49d5416a")  # snapshot stabile
    if run_cluster:
        os.environ["TORCH_HOME"] = "/cluster/home/niacobone/torch_cache"
        try:
            _torch_hub.set_dir(os.environ["TORCH_HOME"])
            print(f"[INFO] Torch hub cache dir set to {_torch_hub.get_dir()}")
        except Exception:
            pass
        OUT_DIR = "/cluster/work/igp_psr/niacobone/distillation/output"
        BASE_DIR = "/cluster/scratch/niacobone/distillation/dataset"
        SAM2_PATH = "/cluster/scratch/niacobone/sam2/checkpoints/sam2.1_hiera_large.pt"
        CONFIG_JSON_PATH = "/cluster/scratch/niacobone/.cache/huggingface/hub/models--facebook--map-anything/snapshots/562de9ff7077addd5780415661c5fb031eb8003e"
        # CONFIG_JSON_PATH = f"/cluster/scratch/niacobone/.cache/huggingface/hub/models--facebook--map-anything/snapshots/{STABLE_REV}"
    else:
        OUT_DIR = "/scratch2/nico/distillation/output"
        BASE_DIR = "/scratch2/nico/distillation/dataset"
        SAM2_PATH = "/scratch2/nico/sam2/checkpoints/sam2.1_hiera_large.pt"
        CONFIG_JSON_PATH = "/scratch/.cache/niacobone/huggingface/hub/models--facebook--map-anything/snapshots/562de9ff7077addd5780415661c5fb031eb8003e"
        # CONFIG_JSON_PATH = f"/cluster/scratch/niacobone/.cache/huggingface/hub/models--facebook--map-anything/snapshots/{STABLE_REV}"
    
    # CONFIG_JSON_PATH = os.path.join(CONFIG_JSON_PATH, "config.json")

    # Usa args.dataset
    DATASET = args.dataset  # "coco2017" o "ETH3D"

    if DATASET == "coco2017":
        TRAIN_SPLIT = "train2017"
        VAL_SPLIT = "val2017"
    else:
        TRAIN_SPLIT = "train"
        VAL_SPLIT = "val"

    IMAGES_DIRNAME = "images"
    FEATURES_DIRNAME = "features"

    BASE_DIR = BASE_DIR + f"/{DATASET}"
    TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, IMAGES_DIRNAME, TRAIN_SPLIT)
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, IMAGES_DIRNAME, VAL_SPLIT)
    TRAIN_FEATURES_DIR = os.path.join(BASE_DIR, FEATURES_DIRNAME, TRAIN_SPLIT)
    VAL_FEATURES_DIR = os.path.join(BASE_DIR, FEATURES_DIRNAME, VAL_SPLIT)
    CONFIG_JSON_PATH = os.path.join(CONFIG_JSON_PATH, "config.json")

    print(f"[INFO] Using TRAIN_IMAGES_DIR: {TRAIN_IMAGES_DIR}")
    print(f"[INFO] Using VAL_IMAGES_DIR: {VAL_IMAGES_DIR}")
    print(f"[INFO] Using TRAIN_FEATURES_DIR: {TRAIN_FEATURES_DIR}")
    print(f"[INFO] Using VAL_FEATURES_DIR: {VAL_FEATURES_DIR}")

# ==================== Dataset Classes ====================
class DistillationDataset(Dataset):
    """
    Dataset per la distillazione: supporta Single-View (lista piatta) e Multi-View (scene folders).
    """
    
    def __init__(
        self,
        image_dir: str,
        features_dir: Optional[str] = None,
        teacher_extractor: Optional[callable] = None, # si può rimuovere
        image_paths: Optional[List[str]] = None,
        transform=None,
        multi_view_mode: bool = False,
        max_views_per_scene: int = 6,
        split: str = "train",
    ):
        self.image_dir = Path(image_dir)
        self.features_dir = Path(features_dir) if features_dir else None
        self.teacher_extractor = teacher_extractor # si può rimuovere
        self.transform = transform
        self.multi_view_mode = multi_view_mode
        self.max_views_per_scene = max_views_per_scene
        self.is_train = "train" in split.lower()
        
        # Validation
        if self.features_dir is None and self.teacher_extractor is None:
            raise ValueError("Either features_dir or teacher_extractor must be provided")
        
        self.mode = "precomputed" if self.features_dir else "online"
        
        # Discovery dei samples
        self.samples = [] 
        if image_paths is not None:
            # Se paths forniti manualmente (es. debug), usiamo quelli.
            # In multi_view, si assume che image_paths sia una lista di liste o gestita esternamente,
            # ma per semplicità qui manteniamo la logica base o appiattita.
            self.samples = image_paths
        else:
            if self.multi_view_mode:
                # --- LOGICA MULTI-VIEW (SCENE) ---
                # Ogni "sample" è una lista di path immagini appartenenti alla stessa scena
                scene_dirs = sorted([d for d in self.image_dir.iterdir() if d.is_dir()])
                for scene in scene_dirs:
                    views = sorted([
                        str(f) for f in scene.iterdir() 
                        if self._is_image_file(f.name)
                    ])
                    if len(views) > 0:
                        self.samples.append(views) # List[str]
                print(f"[Dataset] Mode: MULTI-VIEW (Scenes) | Split: {split} | Max Views: {self.max_views_per_scene}")
            else:
                # --- LOGICA SINGLE-VIEW ---
                # Ogni "sample" è una stringa (path immagine)
                self.samples = sorted([
                    str(self.image_dir / f)
                    for f in os.listdir(self.image_dir)
                    if self._is_image_file(f)
                ])
                print(f"[Dataset] Mode: SINGLE-VIEW (Images)")
                print(f"[Dataset] Found {len(self.samples)} images in {image_dir}")

        if self.mode == "precomputed":
            print(f"[Dataset] Using pre-computed features from {self.features_dir}")
        else:
            print(f"[Dataset] Using online feature extraction with teacher model")
    
    @staticmethod
    def _is_image_file(name: str) -> bool:
        return name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_single_feature(self, img_path: str) -> torch.Tensor:
        """Helper per caricare la feature di una singola immagine"""
        rel_path = Path(img_path).relative_to(self.image_dir)
        # La feature ha la stessa struttura di cartelle dell'immagine ma estensione .pt
        feat_path = self.features_dir / rel_path.with_suffix(".pt")
        
        if not feat_path.exists():
             # Fallback: prova flat se non trova struttura ricorsiva
            feat_path_flat = self.features_dir / Path(img_path).stem
            feat_path_flat = feat_path_flat.with_suffix(".pt")
            if not feat_path_flat.exists():
                raise FileNotFoundError(f"Teacher features not found: {feat_path}")
            feat_path = feat_path_flat

        teacher_feat = torch.load(feat_path, map_location="cpu", weights_only=False)
        if teacher_feat.ndim == 4 and teacher_feat.shape[0] == 1:
            teacher_feat = teacher_feat.squeeze(0)
        return teacher_feat # (C, H, W)

    def __getitem__(self, idx: int) -> Dict:
        """
        Ritorna un dizionario contenente paths, features e (opzionalmente) PIL images.
        Se Multi-View: ritorna le liste per l'intera scena.
        """
        sample = self.samples[idx] # Può essere str (single) o List[str] (multi)
        
        # Normalizziamo tutto a liste per gestire single/multi uniformemente qui dentro
        img_paths = sample if isinstance(sample, list) else [sample]

        # ----- LOGICA DI SUBSAMPLING -----
        if self.multi_view_mode and len(img_paths) > self.max_views_per_scene:
            if self.is_train:
                # TRAINING: Random Sampling (Augmentation)
                # sorted() dopo sample assicura che l'ordine temporale/numerico sia mantenuto
                img_paths = sorted(random.sample(img_paths, self.max_views_per_scene))
            else:
                # VALIDATION: Deterministic Slicing (Consistency)
                # Prende sempre le prime N view in ordine alfabetico/numerico
                img_paths = sorted(img_paths)[:self.max_views_per_scene]
        # -----------------------------------
        
        try:
            teacher_feats_list = []
            pil_images_list = []
            
            if self.mode == "precomputed":
                for p in img_paths:
                    feat = self._load_single_feature(p)
                    teacher_feats_list.append(feat)
                # Stack features: (N_views, C, H, W)
                teacher_features = torch.stack(teacher_feats_list, dim=0)
                pil_images = None
            else:
                # Online mode: carica immagini PIL
                from PIL import Image
                for p in img_paths:
                    pil_images_list.append(Image.open(p).convert("RGB"))
                teacher_features = None
                pil_images = pil_images_list

            return {
                "image_paths": img_paths,          # List[str] (1 o N)
                "teacher_features": teacher_features, # Tensor (N,C,H,W) o None
                "pil_images": pil_images,          # List[PIL] o None
            }

        except Exception as e:
            print(f"[WARN] Error loading sample idx {idx}: {e}")
            # Logica di retry semplificata: solleva errore per ora, 
            # in produzione si può implementare il retry sul sample successivo
            raise e

def collate_fn_distillation(batch: List[Dict]) -> Dict:
    """
    Gestisce il batching.
    Nota: Se batch_size=1 (1 scena), 'batch' è una lista di 1 elemento (il dizionario della scena).
    """
    # batch è una lista di dizionari ritornati da __getitem__
    # Esempio batch_size=1, multi-view:
    # batch = [ {"image_paths": [v1, v2], "teacher_features": Tensor(2,C,H,W), ...} ]
    
    all_image_paths = []
    all_teacher_feats = []
    all_pil_images = []
    
    has_features = batch[0]["teacher_features"] is not None
    has_pil = batch[0]["pil_images"] is not None
    
    for item in batch:
        # Estendiamo le liste (flattening delle scene nel batch se batch_size > 1)
        # Se batch_size=1, stiamo semplicemente prendendo la lista della singola scena.
        all_image_paths.extend(item["image_paths"])
        
        if has_features:
            all_teacher_feats.append(item["teacher_features"])
        
        if has_pil:
            all_pil_images.extend(item["pil_images"])
            
    # Concateniamo i tensori feature
    if has_features:
        # Ogni item["teacher_features"] è (N_views, C, H, W).
        # cat dim=0 -> (Total_Views_in_Batch, C, H, W)
        teacher_feats_tensor = torch.cat(all_teacher_feats, dim=0)
    else:
        teacher_feats_tensor = None
        
    return {
        "image_paths": all_image_paths,       # Lista piatta di tutte le view nel batch
        "teacher_features": teacher_feats_tensor,
        "pil_images": all_pil_images if has_pil else None,
    }

class TeacherFeatureExtractor:
    """
    Wrapper per estrazione feature SAM2 con gestione memoria efficiente.
    TeacherFeatureExtractor è un wrapper “di alto livello” che contiene self.extractor,
    che è un SAM2FeatureExtractor, che a sua volta wrappa un ImageEncoder composto da trunk=Hiera e neck=FpnNeck.
    """
    def __init__(self, checkpoint_path: str, device: str = "cuda", augment_cfg: Optional[dict] = None):
        from sam2_builder import load_sam2_feature_extractor
        # extractor è un'istanza di SAM2FeatureExtractor che è a sua volta un wrapper che contiene image_encoder (trunk + neck)
        self.extractor = load_sam2_feature_extractor(checkpoint_path, device) 
        self.device = device
        self.augment_cfg = augment_cfg or {}
        self._build_augment_pipelines()
        print(f"[Teacher] Loaded SAM2 feature extractor on {device}")

    def _build_augment_pipelines(self):
        """Costruisce le pipeline di augmentation IDENTICHE a MapAnything ufficiale."""
        if not self.augment_cfg.get("enabled", False):
            self.augment_single = None
            self.augment_shared = None
            return
        
        # Parametri UFFICIALI MapAnything (transform="colorjitter+grayscale+gaublur")
        p_color = 0.75   # ColorJitter al 75%
        p_blur = 0.05    # GaussianBlur al 5%
        p_gray = 0.05    # Grayscale al 5%
        
        def _color_jitter():
            return transforms.ColorJitter(
                brightness=0.3,
                contrast=0.4,
                saturation=0.2,
                hue=0.1
            )
        
        def _gaussian_blur():
            return transforms.GaussianBlur(
                kernel_size=5,
                sigma=(0.1, 1.0)
            )
        
        def _make_pipeline():
            ops = []
            ops.append(transforms.RandomApply([_color_jitter()], p=p_color))
            ops.append(transforms.RandomGrayscale(p=p_gray))
            ops.append(transforms.RandomApply([_gaussian_blur()], p=p_blur))
            return transforms.Compose(ops)
        
        self.augment_shared = _make_pipeline()
        self.augment_single = _make_pipeline()

    @torch.no_grad()
    def __call__(self, pil_images: List, multi_view: bool = False, debug_visualize: bool = False) -> torch.Tensor:
        """Estrae feature con augmentation e opzionale debug visualizzazione."""
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        features = []
        use_aug = self.augment_cfg.get("enabled", False)
        
        # DEBUG: Salva PRIMA augmentation
        if debug_visualize and use_aug:
            debug_dir = Path("debug_augmentation")
            debug_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            for idx, pil_img in enumerate(pil_images):
                pil_img.save(debug_dir / f"{timestamp}_before_{idx:02d}.png")
        
        # Applica augmentation
        if use_aug and multi_view and self.augment_shared is not None:
            scene_seed = random.randint(0, 2**31-1)
            augmented_imgs = []
            for pil_img in pil_images:
                torch.manual_seed(scene_seed)
                random.seed(scene_seed)
                augmented_imgs.append(self.augment_shared(pil_img))
        else:
            augmented_imgs = []
            for pil_img in pil_images:
                if use_aug and self.augment_single is not None:
                    augmented_imgs.append(self.augment_single(pil_img))
                else:
                    augmented_imgs.append(pil_img)
        
        # DEBUG: Salva DOPO augmentation + comparison
        if debug_visualize and use_aug:
            for idx, aug_img in enumerate(augmented_imgs):
                aug_img.save(debug_dir / f"{timestamp}_after_{idx:02d}.png")
            
            n = len(pil_images)
            fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
            if n == 1:
                axes = axes.reshape(2, 1)
            
            for idx in range(n):
                axes[0, idx].imshow(pil_images[idx])
                axes[0, idx].set_title(f"Before #{idx}")
                axes[0, idx].axis('off')
                
                axes[1, idx].imshow(augmented_imgs[idx])
                axes[1, idx].set_title(f"After #{idx}")
                axes[1, idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(debug_dir / f"{timestamp}_comparison.png", dpi=150)
            plt.close()
            print(f"[DEBUG] Salvate immagini in {debug_dir}/")
        
        # Estrai features
        for pil_img in augmented_imgs:
            feat = self.extractor(pil_img)
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat)
            if feat.ndim == 3:
                feat = feat.unsqueeze(0)
            features.append(feat)
        
        return torch.cat(features, dim=0)
    
    def to(self, device):
        """Move extractor to device."""
        self.device = device
        if hasattr(self.extractor, "to"):
            self.extractor.to(device)
        return self


# ==================== SAM2 Decoder Distillation Helpers ====================
def _build_sam_prompt_encoder(
    embed_dim: int = 256,
    image_embedding_size: Tuple[int, int] = (64, 64),
    input_image_size: Tuple[int, int] = (1024, 1024),
    mask_in_chans: int = 16,
) -> PromptEncoder:
    return PromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=image_embedding_size,
        input_image_size=input_image_size,
        mask_in_chans=mask_in_chans,
    )


def _build_sam_mask_decoder(
    embed_dim: int = 256,
    num_multimask_outputs: int = 3,
    pred_obj_scores: bool = False,
    pred_obj_scores_mlp: bool = False,
    use_high_res_features: bool = False,
    use_multimask_token_for_obj_ptr: bool = False,
) -> MaskDecoder:
    transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=embed_dim,
        num_heads=8,
        mlp_dim=2048,
    )
    return MaskDecoder(
        transformer_dim=embed_dim,
        transformer=transformer,
        num_multimask_outputs=num_multimask_outputs,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_high_res_features=use_high_res_features,
        pred_obj_scores=pred_obj_scores,
        pred_obj_scores_mlp=pred_obj_scores_mlp,
        use_multimask_token_for_obj_ptr=use_multimask_token_for_obj_ptr,
    )


def _infer_teacher_decoder_flags_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, bool]:
    """Infer optional MaskDecoder constructor flags from checkpoint keys."""
    use_high_res_features = any(k.startswith("sam_mask_decoder.conv_s0") for k in sd.keys())
    pred_obj_scores = any(k.startswith("sam_mask_decoder.obj_score_token") for k in sd.keys())
    pred_obj_scores_mlp = any(k.startswith("sam_mask_decoder.pred_obj_score_head.layers") for k in sd.keys())
    return {
        "use_high_res_features": bool(use_high_res_features),
        "pred_obj_scores": bool(pred_obj_scores),
        "pred_obj_scores_mlp": bool(pred_obj_scores_mlp),
    }


def load_sam2_teacher_prompt_and_decoder(
    checkpoint_path: str,
    device: torch.device,
    embed_dim: int = 256,
    image_embedding_size: Tuple[int, int] = (64, 64),
    input_image_size: Tuple[int, int] = (1024, 1024),
) -> Tuple[PromptEncoder, MaskDecoder]:
    """Load SAM2 prompt encoder + mask decoder weights from a SAM2 checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt)

    flags = _infer_teacher_decoder_flags_from_state_dict(sd)
    prompt_encoder = _build_sam_prompt_encoder(
        embed_dim=embed_dim,
        image_embedding_size=image_embedding_size,
        input_image_size=input_image_size,
        mask_in_chans=16,
    )
    mask_decoder = _build_sam_mask_decoder(
        embed_dim=embed_dim,
        num_multimask_outputs=3,
        pred_obj_scores=flags["pred_obj_scores"],
        pred_obj_scores_mlp=flags["pred_obj_scores_mlp"],
        use_high_res_features=flags["use_high_res_features"],
        use_multimask_token_for_obj_ptr=False,
    )

    pe_sd = {k.replace("sam_prompt_encoder.", ""): v for k, v in sd.items() if k.startswith("sam_prompt_encoder.")}
    md_sd = {k.replace("sam_mask_decoder.", ""): v for k, v in sd.items() if k.startswith("sam_mask_decoder.")}

    missing_pe, unexpected_pe = prompt_encoder.load_state_dict(pe_sd, strict=False)
    missing_md, unexpected_md = mask_decoder.load_state_dict(md_sd, strict=False)
    if missing_pe or unexpected_pe:
        print(f"[WARN] PromptEncoder load_state_dict: missing={len(missing_pe)} unexpected={len(unexpected_pe)}")
    if missing_md or unexpected_md:
        print(f"[WARN] MaskDecoder load_state_dict: missing={len(missing_md)} unexpected={len(unexpected_md)}")

    prompt_encoder = prompt_encoder.to(device)
    mask_decoder = mask_decoder.to(device)
    prompt_encoder.eval()
    mask_decoder.eval()
    for p in prompt_encoder.parameters():
        p.requires_grad = False
    for p in mask_decoder.parameters():
        p.requires_grad = False

    return prompt_encoder, mask_decoder


@torch.no_grad()
def forward_sam_decoder_no_prompt(
    mask_decoder: MaskDecoder,
    prompt_encoder: PromptEncoder,
    image_embeddings: torch.Tensor,
    multimask_output: bool = True,
    high_res_features: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run MaskDecoder with empty prompts (no points/boxes/masks), mirroring SAM2 defaults."""
    B = image_embeddings.shape[0]
    device = image_embeddings.device
    dtype = image_embeddings.dtype

    # If no points are provided, SAM2 pads with an empty point (label -1)
    point_coords = torch.zeros(B, 1, 2, device=device)
    point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

    sparse_embeddings, dense_embeddings = prompt_encoder(
        points=(point_coords, point_labels),
        boxes=None,
        masks=None,
    )

    image_pe = prompt_encoder.get_dense_pe().to(device=device, dtype=dtype)
    sparse_embeddings = sparse_embeddings.to(dtype=dtype)
    dense_embeddings = dense_embeddings.to(dtype=dtype)

    masks, iou_pred, sam_tokens, obj_scores = mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        repeat_image=False,
        high_res_features=high_res_features,
    )
    return masks, iou_pred, sam_tokens, obj_scores


def decoder_distillation_loss_mse(
    student_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    teacher_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute simple MSE on the 4 decoder outputs: masks, iou_pred, sam_tokens, obj_scores."""
    s_masks, s_iou, s_tokens, s_obj = student_out
    t_masks, t_iou, t_tokens, t_obj = teacher_out

    mse_masks = F.mse_loss(s_masks.float(), t_masks.float())
    mse_iou = F.mse_loss(s_iou.float(), t_iou.float())
    mse_tokens = F.mse_loss(s_tokens.float(), t_tokens.float())
    mse_obj = F.mse_loss(s_obj.float(), t_obj.float())

    total = mse_masks + mse_iou + mse_tokens + mse_obj
    details = {
        "dec_mse_masks": float(mse_masks.detach().cpu().item()),
        "dec_mse_iou": float(mse_iou.detach().cpu().item()),
        "dec_mse_tokens": float(mse_tokens.detach().cpu().item()),
        "dec_mse_obj": float(mse_obj.detach().cpu().item()),
        "dec_loss": float(total.detach().cpu().item()),
    }
    return total, details

# ==================== Loss Functions ====================
class DistillationLoss(torch.nn.Module):
    """
    Loss combinata MSE + (1 - Cosine Similarity) per la distillazione delle feature.
    
    Args:
        mse_weight: peso della componente MSE
        cosine_weight: peso della componente (1 - cosine similarity)
        normalize: se True, normalizza lungo i canali (dim=1) prima del calcolo
    """
    
    def __init__(
        self,
        mse_weight: float = 0.5,
        cosine_weight: float = 0.5,
        normalize: bool = False,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.normalize = normalize
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        mse_type: str = "pixel",
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Calcola la loss di distillazione.
        
        Args:
            student_features: Tensor (B,C,H,W) dallo studente
            teacher_features: Tensor (B,C,H,W) dal teacher
            mse_type: Tipo di MSE ("pixel" o "sample")
        
        Returns:
            loss: valore scalare totale
            loss_details: dizionario con componenti ('mse_loss','cos_loss','cos_sim')
        Note:
            F.cosine_similarity(..., dim=1) produce (B,H,W); qui facciamo .mean() su tutte le posizioni.
        """
        # Optionally normalize
        if self.normalize:
            student_norm = F.normalize(student_features, dim=1)
            teacher_norm = F.normalize(teacher_features, dim=1)
        else:
            student_norm = student_features
            teacher_norm = teacher_features

        cos_map = F.cosine_similarity(student_norm, teacher_norm, dim=1)  # (B, H, W)
        cos_sim_per_image = cos_map.flatten(1).mean(dim=1)  # Media su (H,W) → (B,)
        cos_sim = cos_sim_per_image.mean()  # Media su batch → scalare
        
        cos_loss = 1.0 - cos_sim

        if mse_type == "sample":
            # Calcola MSE sample-wise (media su batch, ma SOMMA su C,H,W)
            mse_per_sample = F.mse_loss(
                student_norm, 
                teacher_norm, 
                reduction='none'  # (B, C, H, W)
            ).mean(dim=(1, 2, 3))  # Media su (C,H,W) → (B,)
            
            mse_loss = mse_per_sample.mean()  # Media su batch → scalare

        elif mse_type == "pixel":
            # Calcola MSE pixel-wise (media su batch, H, W, ma SOMMA su canali C)
            diff = student_norm - teacher_norm
            mse_loss = (diff ** 2).sum(dim=1).mean() # Somma su C, media su H,W
        
        # Combined loss
        total_loss = self.mse_weight * mse_loss + self.cosine_weight * cos_loss
        
        loss_details = {
            "mse_loss": mse_loss.item(),
            "cos_loss": cos_loss.item(),
            "cos_sim": cos_sim.item(),
        }
        
        return total_loss, loss_details

# ==================== Data Loaders ====================
def build_distillation_dataloader(
    image_dir: str,
    features_dir: Optional[str] = None,
    teacher_extractor: Optional[TeacherFeatureExtractor] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    image_paths: Optional[List[str]] = None,
    pin_memory: bool = True,
    distributed: bool = False,
    multi_view_mode: bool = False,
    split: str = "train",
    max_views_per_scene: int = 6,
) -> DataLoader:
    """
    Build a DataLoader for distillation training/validation.
    
    Args:
        image_dir: Directory containing images
        features_dir: Directory containing teacher features (None for online mode)
        teacher_extractor: TeacherFeatureExtractor instance (required if features_dir=None)
        batch_size: Batch size per GPU
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the dataset (ignored if distributed=True)
        image_paths: Optional list of specific image paths to use
        pin_memory: Whether to use pinned memory
        distributed: Whether to use DistributedSampler for multi-GPU training
    
    Returns:
        DataLoader per distillazione con supporto pre-computed/online
    """
    dataset = DistillationDataset(
        image_dir=image_dir,
        features_dir=features_dir,
        teacher_extractor=teacher_extractor,
        image_paths=image_paths,
        multi_view_mode=multi_view_mode,
        split=split,
        max_views_per_scene=max_views_per_scene,
    )
    
    sampler = None
    if distributed:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=shuffle,
        )
        shuffle = False
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_distillation,
        drop_last=shuffle and sampler is None,
    )
    
    return loader

# ==================== Training Functions ====================
def forward_pass_distillation_unified(
    model: torch.nn.Module,
    image_paths: List[str],
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: str = "bf16",
    process_individually: bool = True,
    use_encoder_features: bool = False,
) -> torch.Tensor:
    """
    Forward pass unificato per MapAnything distillation.
    Supporta sia single-view batch-safe che multi-view con cross-attention.
    
    Args:
        model: MapAnything model
        image_paths: Lista di path immagini (flat list)
        device: Device CUDA
        use_amp: Se usare mixed precision
        amp_dtype: Tipo AMP ("bf16" o "fp16")
        process_individually: Se True, ogni immagine è processata separatamente
                             (NO cross-attention, single-view batch-safe).
                             Se False, tutte le immagini sono caricate insieme
                             (cross-attention applicato, multi-view).
    
    Returns:
        torch.Tensor: Student features (B, C, H, W) dove B = len(image_paths)
    
    Examples:
        >>> # Single-view batch-safe (immagini indipendenti)
        >>> features = forward_pass_distillation_unified(
        ...     model, ["img1.jpg", "img2.jpg"], device,
        ...     process_individually=True
        ... )
        >>> features.shape  # (2, 256, 64, 64) - NO cross-attention
        
        >>> # Multi-view (gruppo di views della stessa scena)
        >>> features = forward_pass_distillation_unified(
        ...     model, ["scene1_v0.jpg", "scene1_v1.jpg"], device,
        ...     process_individually=False
        ... )
        >>> features.shape  # (2, 256, 64, 64) - CON cross-attention
    """
    
    amp_dtype_torch = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    
    if process_individually:
        # ========== SINGLE-VIEW BATCH-SAFE ==========
        # Processa ogni immagine separatamente per evitare cross-attention
        all_features = []
        
        for img_path in image_paths:
            # Carica singola immagine come lista di 1 view
            views = load_images([img_path])
            
            # Forward con AMP
            with torch.autocast("cuda", enabled=use_amp, dtype=amp_dtype_torch):
                # Sposta su device DENTRO autocast per fix dtype mismatch
                for v in views:
                    img = v.get("img")
                    if isinstance(img, torch.Tensor):
                        v["img"] = img.to(device, non_blocking=True)
                
                # MapAnything vede SINGOLA VIEW → NO cross-attention
                _ = model(views, memory_efficient_inference=False, use_encoder_features=use_encoder_features)
            
            # Estrai feature dalla view 0 (unica view)
            base_model = model.module if hasattr(model, "module") else model
            student_features_single = getattr(base_model, "_last_feat2_8x", None)

            ############## DEBUG VISUALIZZAZIONE PCA ##############
            # encoder_features = getattr(base_model, "_last_encoder_features", None)
            # print(f"[DEBUG] student_features_single shape: {student_features_single.shape if student_features_single is not None else 'None'}")
            # print(f"[DEBUG] encoder_features shape: {encoder_features.shape if encoder_features is not None else 'None'}")
            # # Resize encoder features to match student spatial resolution
            # encoder_resized = F.interpolate(
            #     encoder_features,
            #     size=student_features_single.shape[-2:],  # (64, 64)
            #     mode="bilinear",
            #     align_corners=False
            # )

            # self_dense_input = getattr(base_model, "_nico_dense_head_inputs", None)

            # self_dense_input[0] = F.interpolate(
            #     self_dense_input[0],
            #     size=student_features_single.shape[-2:],  # (64, 64)
            #     mode="bilinear",
            #     align_corners=False
            # )
            # self_dense_input[1] = F.interpolate(
            #     self_dense_input[1],
            #     size=student_features_single.shape[-2:],  # (64, 64)
            #     mode="bilinear",
            #     align_corners=False
            # )
            # self_dense_input[2] = F.interpolate(
            #     self_dense_input[2],
            #     size=student_features_single.shape[-2:],  # (64, 64)
            #     mode="bilinear",
            #     align_corners=False
            # )
            # self_dense_input[3] = F.interpolate(
            #     self_dense_input[3],
            #     size=student_features_single.shape[-2:],  # (64, 64)
            #     mode="bilinear",
            #     align_corners=False
            # )
            
            # Project encoder channels (1024) to student channels (256)
            # Simple average pooling over channel groups
            # B, C_enc, H, W = encoder_resized.shape
            # C_student = student_features_single.shape[1]
            # group_size = C_enc // C_student  # 1024 // 256 = 4
            
            # encoder_projected = encoder_resized.reshape(B, C_student, group_size, H, W).mean(dim=2)
            # dense_0 = self_dense_input[0].reshape(B, C_student, group_size, H, W).mean(dim=2)
            # dense_1 = self_dense_input[1].reshape(B, C_student, group_size, H, W).mean(dim=2)
            # dense_2 = self_dense_input[2].reshape(B, C_student, group_size, H, W).mean(dim=2)
            # dense_3 = self_dense_input[3].reshape(B, C_student, group_size, H, W).mean(dim=2)

            # Now encoder_projected is (1, 256, 64, 64) - same shape as student
            # save_pca_visualizations(
            #         student_features=student_features_single,
            #         teacher_features=encoder_projected,
            #         image_paths=image_paths,
            #         epoch=1,
            #         output_dir="/scratch2/nico/distillation/output/test_dino",
            #     )
            # print("fatto 1")
            # save_pca_visualizations(
            #         student_features=dense_0,
            #         teacher_features=dense_1,
            #         image_paths=image_paths,
            #         epoch=2,
            #         output_dir="/scratch2/nico/distillation/output/test_dino",
            #     )
            # print("fatto 2")
            # save_pca_visualizations(
            #         student_features=dense_2,
            #         teacher_features=dense_3,
            #         image_paths=image_paths,
            #         epoch=3,
            #         output_dir="/scratch2/nico/distillation/output/test_dino",
            #     )
            # print("fatto 3")

            ######################################################

            if student_features_single is None:
                raise KeyError(
                    "Student features not found on model (_last_feat2_8x). "
                    "Ensure dpt_feature_head_2 is present and forward populates this attribute."
                )
            
            all_features.append(student_features_single)
        
        # Concatena feature di tutte le immagini
        return torch.cat(all_features, dim=0)  # (B, C, H, W)
    
    else:
        # ========== MULTI-VIEW ==========
        # Carica tutte le immagini insieme (cross-attention applicato)
        views = load_images(image_paths)
        
        # Forward con AMP
        with torch.autocast("cuda", enabled=use_amp, dtype=amp_dtype_torch):
            # Sposta su device DENTRO autocast
            for v in views:
                img = v.get("img")
                if isinstance(img, torch.Tensor):
                    v["img"] = img.to(device, non_blocking=True)
            
            # MapAnything vede N VIEWS → cross-attention tra tutte
            _ = model(views, memory_efficient_inference=False, use_encoder_features=use_encoder_features)
        
        # Estrai feature (già tutte in un batch)
        base_model = model.module if hasattr(model, "module") else model
        student_features = getattr(base_model, "_last_feat2_8x", None)
        ############## DEBUG VISUALIZZAZIONE PCA ##############
        # encoder_features = getattr(base_model, "_last_encoder_features", None)
        # save_pca_visualizations(
        #         student_features=student_features,
        #         teacher_features=encoder_features,
        #         image_paths=image_paths,
        #         epoch=1,
        #         output_dir="/scratch2/nico/distillation/output/test_dino",
        #     )
        ######################################################
        if student_features is None:
            raise KeyError("Student features not found (_last_feat2_8x)")
        
        return student_features  # (B, C, H, W) dove B = len(image_paths)

def train_one_epoch_distillation(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    teacher_extractor: Optional[TeacherFeatureExtractor] = None,
    sam_prompt_encoder_teacher: Optional[PromptEncoder] = None,
    sam_mask_decoder_teacher: Optional[MaskDecoder] = None,
) -> Dict:
    """
    Train the model for one epoch on distillation task.
    
    Args:
        model: MapAnything model with dpt_feature_head_2
        criterion: DistillationLoss instance
        data_loader: DataLoader providing image paths and teacher features
        optimizer: Optimizer
        device: Device to run on
        epoch: Current epoch number
        args: Configuration namespace
        teacher_extractor: TeacherFeatureExtractor for online mode (None if precomputed)
    
    Returns:
        Dictionary of averaged training metrics
    """
    model.train(True)
    metric_logger = train_tools.MetricLogger(delimiter=" | ")
    metric_logger.add_meter("lr", train_tools.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Distillation Epoch: [{epoch}]"
    
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    # Detect mode from first batch
    mode = None
    
    total_samples = 0
    sum_loss = 0.0
    sum_mse = 0.0
    sum_cos = 0.0
    sum_cos_sim = 0.0
    sum_dec_loss = 0.0
    sum_dec_mse_masks = 0.0
    sum_dec_mse_iou = 0.0
    sum_dec_mse_tokens = 0.0
    sum_dec_mse_obj = 0.0
    sum_mean_diff = 0.0
    sum_std_diff = 0.0
    
    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        # Auto-detect mode from first batch
        if mode is None:
            mode = "precomputed" if batch["teacher_features"] is not None else "online"
            print(f"[Train] Feature extraction mode: {mode.upper()}")
        
        epoch_f = epoch + data_iter_step / max(1, len(data_loader))
        
        # Get data
        image_paths = batch["image_paths"]
        
        # Extract or load teacher features
        if mode == "precomputed":
            teacher_features = batch["teacher_features"].to(device, non_blocking=True)
        else:  # online
            if teacher_extractor is None:
                raise ValueError("teacher_extractor required for online mode but not provided")
            
            pil_images = batch["pil_images"]
            with torch.no_grad():
                # PASSA multi_view per coerenza intra-scena
                # teacher_features = teacher_extractor(pil_images, multi_view=args.multi_view_mode).to(device, non_blocking=True)
                teacher_features = teacher_extractor(
                    pil_images, 
                    multi_view=args.multi_view_mode,
                    # debug_visualize=(data_iter_step < 3 and epoch == 0)  # ← Primi 3 batch
                ).to(device, non_blocking=True)
        
        # Forward pass to get student features
        student_features = forward_pass_distillation_unified(
            model=model,
            image_paths=image_paths,
            device=device,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
            process_individually=not args.multi_view_mode,
            use_encoder_features=args.use_encoder_features,
        )
        
        # Resize student features to match teacher resolution if needed
        if student_features.shape[-2:] != teacher_features.shape[-2:]:
            H, W = teacher_features.shape[-2:]
            student_features = F.interpolate(
                student_features,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        # Compute encoder-feature distillation loss
        loss, loss_details = criterion(student_features, teacher_features, mse_type=args.mse_type)

        # Optional: Decoder distillation (online)
        dec_total = None
        dec_details: Dict[str, float] = {}
        if getattr(args, "distill_decoder", False):
            if sam_prompt_encoder_teacher is None or sam_mask_decoder_teacher is None:
                raise ValueError("distill_decoder=True but teacher prompt/decoder not provided")
            base_model = model.module if hasattr(model, "module") else model
            if not hasattr(base_model, "sam2_mask_decoder_student"):
                raise AttributeError("Model missing sam2_mask_decoder_student; ensure it is attached in distill()")
            student_decoder: MaskDecoder = base_model.sam2_mask_decoder_student

            # Forward teacher/student decoders with empty prompts
            with torch.no_grad():
                teacher_out = forward_sam_decoder_no_prompt(
                    mask_decoder=sam_mask_decoder_teacher,
                    prompt_encoder=sam_prompt_encoder_teacher,
                    image_embeddings=teacher_features,
                    multimask_output=getattr(args, "decoder_multimask_output", True),
                    high_res_features=None,
                )
            student_out = forward_sam_decoder_no_prompt(
                mask_decoder=student_decoder,
                prompt_encoder=sam_prompt_encoder_teacher,
                image_embeddings=student_features,
                multimask_output=getattr(args, "decoder_multimask_output", True),
                high_res_features=None,
            )
            dec_total, dec_details = decoder_distillation_loss_mse(student_out, teacher_out)
            loss = loss + float(getattr(args, "decoder_loss_weight", 1.0)) * dec_total
        mse_value = float(loss_details.get("mse_loss", 0.0))
        cos_value = float(loss_details.get("cos_loss", 0.0))
        cos_sim_value = float(loss_details.get("cos_sim", 0.0))
        dec_loss_value = float(dec_details.get("dec_loss", 0.0))

        try:
            md, sd, cs = mean_std_difference(student_features, teacher_features)
            md = float(md)
            sd = float(sd)
            cs = float(cs)
        except Exception:
            md = sd = 0.0
            cs = cos_sim_value

        loss_value = loss.detach().cpu().item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training", flush=True)
            sys.exit(1)

        # W&B batch-level logging
        is_main_process = (train_tools.get_rank() == 0)
        if (
            args.use_wandb
            and WANDB_AVAILABLE
            and is_main_process
            and (wandb is not None)
            and (getattr(wandb, "run", None) is not None)
        ):
            if data_iter_step % getattr(args, "log_freq", 100) == 0:
                wandb.log(
                    {
                        "train/loss": float(loss_value),
                        "train/mse_loss": float(mse_value),
                        "train/cos_loss": float(cos_value),
                        "train/cos_sim": float(cos_sim_value),
                        "train/decoder_loss": float(dec_loss_value),
                        "train/dec_mse_masks": float(dec_details.get("dec_mse_masks", 0.0)),
                        "train/dec_mse_iou": float(dec_details.get("dec_mse_iou", 0.0)),
                        "train/dec_mse_tokens": float(dec_details.get("dec_mse_tokens", 0.0)),
                        "train/dec_mse_obj": float(dec_details.get("dec_mse_obj", 0.0)),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "epoch_progress": epoch_f,
                    }
                )
        
        # Gradient Accumulation
        loss /= accum_iter
        loss.backward()

        if (data_iter_step + 1) % accum_iter == 0:
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            # Optional: interactive LR inspection with pdb (halts execution)
            # if getattr(args, "debug_pdb_lr", False):
            #     import pdb
            #     pdb.set_trace()
            optimizer.zero_grad()

        # Accumulate weighted sums
        batch_size = student_features.shape[0]
        total_samples += batch_size
        sum_loss += loss_value * batch_size
        sum_mse += mse_value * batch_size
        sum_cos += cos_value * batch_size
        sum_cos_sim += cos_sim_value * batch_size
        # decoder loss is already included in loss_value; track separately via metric_logger below
        if getattr(args, "distill_decoder", False) and dec_details:
            sum_dec_loss += float(dec_details.get("dec_loss", 0.0)) * batch_size
            sum_dec_mse_masks += float(dec_details.get("dec_mse_masks", 0.0)) * batch_size
            sum_dec_mse_iou += float(dec_details.get("dec_mse_iou", 0.0)) * batch_size
            sum_dec_mse_tokens += float(dec_details.get("dec_mse_tokens", 0.0)) * batch_size
            sum_dec_mse_obj += float(dec_details.get("dec_mse_obj", 0.0)) * batch_size
        sum_mean_diff += md * batch_size
        sum_std_diff += sd * batch_size

        # Clean up
        del loss, student_features, teacher_features
        
        # Update metrics
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss_value, **loss_details)
        if getattr(args, "distill_decoder", False) and dec_details:
            metric_logger.update(decoder_loss=dec_loss_value, **dec_details)

    # Return averaged stats
    denom = max(1, total_samples)
    results = {
        "loss_mean": sum_loss / denom,
        "mse_loss_mean": sum_mse / denom,
        "cos_loss_mean": sum_cos / denom,
        "cos_sim_mean": sum_cos_sim / denom,
        "dec_loss_mean": sum_dec_loss / denom,
        "dec_mse_masks_mean": sum_dec_mse_masks / denom,
        "dec_mse_iou_mean": sum_dec_mse_iou / denom,
        "dec_mse_tokens_mean": sum_dec_mse_tokens / denom,
        "dec_mse_obj_mean": sum_dec_mse_obj / denom,
        "mean_diff": sum_mean_diff / denom,
        "std_diff": sum_std_diff / denom,
        "lr": optimizer.param_groups[0]["lr"],
        "samples": total_samples,
    }
    return results

@torch.no_grad()
def validate_one_epoch_distillation(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    args,
    teacher_extractor: Optional[TeacherFeatureExtractor] = None,
    sam_prompt_encoder_teacher: Optional[PromptEncoder] = None,
    sam_mask_decoder_teacher: Optional[MaskDecoder] = None,
) -> Dict:
    """
    Validate the model for one epoch on distillation task.
    
    Args:
        model: MapAnything model with dpt_feature_head_2
        criterion: DistillationLoss instance
        data_loader: DataLoader providing validation data
        device: Device to run on
        epoch: Current epoch number
        args: Configuration namespace
        teacher_extractor: TeacherFeatureExtractor for online mode
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    metric_logger = train_tools.MetricLogger(delimiter=" | ")
    metric_logger.meters = defaultdict(lambda: train_tools.SmoothedValue(window_size=int(1e6)))
    header = f"Distillation Validation: [{epoch}]"
    
    mode = None
    total_samples = 0
    sum_loss = 0.0
    sum_mse = 0.0
    sum_cos = 0.0
    sum_cos_sim = 0.0
    sum_dec_loss = 0.0
    sum_dec_mse_masks = 0.0
    sum_dec_mse_iou = 0.0
    sum_dec_mse_tokens = 0.0
    sum_dec_mse_obj = 0.0
    sum_mean_diff = 0.0
    sum_std_diff = 0.0

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        # Auto-detect mode
        if mode is None:
            mode = "precomputed" if batch["teacher_features"] is not None else "online"
            print(f"[Val] Feature extraction mode: {mode.upper()}")
        
        image_paths = batch["image_paths"]
        
        # Extract or load teacher features
        if mode == "precomputed":
            teacher_features = batch["teacher_features"].to(device, non_blocking=True)
        else:
            pil_images = batch["pil_images"]
            teacher_features = teacher_extractor(pil_images, multi_view=args.multi_view_mode).to(device, non_blocking=True)
        
        # Forward pass
        student_features = forward_pass_distillation_unified(
            model=model,
            image_paths=image_paths,
            device=device,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
            process_individually=not args.multi_view_mode,
            use_encoder_features=args.use_encoder_features,
        )
        
        # Resize if needed
        if student_features.shape[-2:] != teacher_features.shape[-2:]:
            H, W = teacher_features.shape[-2:]
            student_features = F.interpolate(
                student_features,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
        
        # Compute encoder-feature distillation loss
        loss, loss_details = criterion(student_features, teacher_features, mse_type=args.mse_type)

        # Optional: Decoder distillation
        dec_total = None
        dec_details: Dict[str, float] = {}
        if getattr(args, "distill_decoder", False):
            if sam_prompt_encoder_teacher is None or sam_mask_decoder_teacher is None:
                raise ValueError("distill_decoder=True but teacher prompt/decoder not provided")
            base_model = model.module if hasattr(model, "module") else model
            if not hasattr(base_model, "sam2_mask_decoder_student"):
                raise AttributeError("Model missing sam2_mask_decoder_student; ensure it is attached in distill()")
            student_decoder: MaskDecoder = base_model.sam2_mask_decoder_student

            teacher_out = forward_sam_decoder_no_prompt(
                mask_decoder=sam_mask_decoder_teacher,
                prompt_encoder=sam_prompt_encoder_teacher,
                image_embeddings=teacher_features,
                multimask_output=getattr(args, "decoder_multimask_output", True),
                high_res_features=None,
            )
            student_out = forward_sam_decoder_no_prompt(
                mask_decoder=student_decoder,
                prompt_encoder=sam_prompt_encoder_teacher,
                image_embeddings=student_features,
                multimask_output=getattr(args, "decoder_multimask_output", True),
                high_res_features=None,
            )
            dec_total, dec_details = decoder_distillation_loss_mse(student_out, teacher_out)
            loss = loss + float(getattr(args, "decoder_loss_weight", 1.0)) * dec_total
        loss_value = loss.detach().cpu().item()
        mse_value = float(loss_details.get("mse_loss", 0.0))
        cos_value = float(loss_details.get("cos_loss", 0.0))
        cos_sim_value = float(loss_details.get("cos_sim", 0.0))
        dec_loss_value = float(dec_details.get("dec_loss", 0.0))

        try:
            md, sd, cs = mean_std_difference(student_features, teacher_features)
            md = float(md)
            sd = float(sd)
            cs = float(cs)
        except Exception:
            md = sd = 0.0
            cs = cos_sim_value

        metric_logger.update(loss=loss_value, mse_loss=mse_value, cos_loss=cos_value, cos_sim=cos_sim_value)
        if getattr(args, "distill_decoder", False) and dec_details:
            metric_logger.update(decoder_loss=dec_loss_value, **dec_details)
        
        # Salva visualizzazioni se richiesto
        if args.save_visualizations and batch_idx == 0:
            save_pca_visualizations(
                student_features=student_features,
                teacher_features=teacher_features,
                image_paths=image_paths,
                epoch=epoch,
                output_dir=args.output_dir,
            )
        
        # Accumulate
        batch_size = student_features.shape[0]
        total_samples += batch_size
        sum_loss += loss_value * batch_size
        sum_mse += mse_value * batch_size
        sum_cos += cos_value * batch_size
        sum_cos_sim += cos_sim_value * batch_size
        if getattr(args, "distill_decoder", False) and dec_details:
            sum_dec_loss += float(dec_details.get("dec_loss", 0.0)) * batch_size
            sum_dec_mse_masks += float(dec_details.get("dec_mse_masks", 0.0)) * batch_size
            sum_dec_mse_iou += float(dec_details.get("dec_mse_iou", 0.0)) * batch_size
            sum_dec_mse_tokens += float(dec_details.get("dec_mse_tokens", 0.0)) * batch_size
            sum_dec_mse_obj += float(dec_details.get("dec_mse_obj", 0.0)) * batch_size
        sum_mean_diff += md * batch_size
        sum_std_diff += sd * batch_size

        del student_features, teacher_features

    denom = max(1, total_samples)
    results = {
        "loss_mean": sum_loss / denom,
        "mse_loss_mean": sum_mse / denom,
        "cos_loss_mean": sum_cos / denom,
        "cos_sim_mean": sum_cos_sim / denom,
        "dec_loss_mean": sum_dec_loss / denom,
        "dec_mse_masks_mean": sum_dec_mse_masks / denom,
        "dec_mse_iou_mean": sum_dec_mse_iou / denom,
        "dec_mse_tokens_mean": sum_dec_mse_tokens / denom,
        "dec_mse_obj_mean": sum_dec_mse_obj / denom,
        "mean_diff": sum_mean_diff / denom,
        "std_diff": sum_std_diff / denom,
        "samples": total_samples,
        "loss_avg": sum_loss / denom,
    }
    return results

# ==================== Visualization Functions ====================
def save_pca_visualizations(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    image_paths: List[str],
    epoch: int,
    output_dir: str,
):
    """
    Salva visualizzazioni affiancate basate su PCA di (studente | immagine | teacher).
    
    Uses nico.utils.create_student_original_teacher_side_by_side which handles:
    - PCA conversion of features
    - Loading original image
    - Creating side-by-side composite (student | original | teacher)
    - Saving to disk
    
    Args:
        student_features: (B, C, H, W) student features
        teacher_features: (B, C, H, W) teacher features
        image_paths: List of image paths for this batch
        epoch: Current epoch number
        output_dir: Output directory for saving visualizations
    """
    # Cartella di output per le visualizzazioni
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Move to CPU
    student_cpu = student_features.detach().cpu()  # (B, C, H, W)
    teacher_cpu = teacher_features.detach().cpu()  # (B, C, H, W)
    
    B = student_cpu.shape[0]
    
    for batch_idx in range(B):
        img_path = image_paths[batch_idx]
        
        # Extract single image features (keep batch dimension for nico.utils compatibility)
        student_single = student_cpu[batch_idx:batch_idx+1]  # (1, C, H, W)
        teacher_single = teacher_cpu[batch_idx:batch_idx+1]  # (1, C, H, W)
        
        # Create and save side-by-side visualization using nico.utils function
        # This function handles PCA, image loading, compositing, and saving
        try:
            create_student_original_teacher_side_by_side(
                student_embeddings=student_single,
                teacher_embeddings=teacher_single,
                img_path=img_path,
                epoch=epoch,
                output_heatmaps=str(viz_dir),
                is_overfit_image=False,  # Dynamic PCA basis (not saved/loaded from disk)
            )
            student_save_path = Path(str(viz_dir)) / "student"
            student_save_path.mkdir(parents=True, exist_ok=True)
            teacher_save_path = Path(str(viz_dir)) / "teacher"
            teacher_save_path.mkdir(parents=True, exist_ok=True)
            print(f"Image path: {img_path}")

            # Ensure tensors are detached, cloned, and contiguous before saving
            student_single = student_single.detach().cpu().contiguous().clone()
            teacher_single = teacher_single.detach().cpu().contiguous().clone()
            img_basename = Path(img_path).stem  # Es: "000000544826"
            torch.save(student_single, student_save_path / f"{epoch}_{img_basename}.pt")
            torch.save(teacher_single, teacher_save_path / f"{epoch}_{img_basename}.pt")
        except Exception as e:
            print(f"[WARN] Failed to create PCA visualization for {img_path}: {e}")
            continue
    
    print(f"[VIZ] Saved {B} PCA visualizations to {viz_dir}")

def log_param_status(model, max_print=None):
    trainable, frozen = [], []
    for name, p in model.named_parameters():
        (trainable if p.requires_grad else frozen).append(name)

    print("\n" + "="*80)
    print("DETAILED PARAMETER STATUS")
    print("="*80)
    print(f"Trainable entries: {len(trainable)}")
    print(f"Frozen entries:    {len(frozen)}")

    def _dump(title, items):
        print(title)
        limit = len(items) if max_print is None else min(len(items), max_print)
        for n in sorted(items)[:limit]:
            print(f"  {n}")
        if limit < len(items):
            print(f"  ... and {len(items) - limit} more")

    _dump("[TRAINABLE]", trainable)
    _dump("[FROZEN]", frozen)
    print("="*80)

# ==================== Main Training Loop ====================
def distill(args):
    """
    Main distillation training function.
    """
    setup_runtime_paths(args)
    train_tools.init_distributed_mode(args.distributed)
    global_rank = train_tools.get_rank()
    
    if not args.output_dir:
        default_run_name = args.wandb_name or datetime.datetime.now().strftime("distill_%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(OUT_DIR, default_run_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = not args.disable_cudnn_benchmark
    
    if run_cluster and getattr(args, "print_freq", 10) < 200:
        args.print_freq = 200

    # ========== INITIALIZE TEACHER EXTRACTOR (if online mode) ==========
    teacher_extractor = None
    if not args.precomputed_features:
        print(f"[INFO] Initializing online teacher feature extractor from {SAM2_PATH}")
        augment_cfg = {
            "enabled": not getattr(args, "no_augmentation", False),
            "p_color_jitter": 0.75,     # 75% probabilità (UFFICIALE MapAnything)
            "p_blur": 0.05,              # 5% probabilità (UFFICIALE, era 0.5!)
            "p_grayscale": 0.05,         # 5% probabilità (UFFICIALE, era 0.2!)
            # NOTA: MapAnything NON usa RandomResizedCrop, rimosso da pipeline
        }
        teacher_extractor = TeacherFeatureExtractor(
            checkpoint_path=SAM2_PATH,
            device=str(device),
            augment_cfg=augment_cfg,
        )
        teacher_extractor.to(device)
    
    # ========== BUILD DATALOADERS ==========
    
    # --- 1. TRAIN DATALOADER ---
    print(f"Building train dataloader from {TRAIN_IMAGES_DIR}")
    train_image_paths = None
    
    # Logica Debug per SINGLE-VIEW: filtriamo la lista delle immagini PRIMA di creare il loader
    if args.debug_max_train_images and not args.multi_view_mode:
        all_imgs = sorted([
            os.path.join(TRAIN_IMAGES_DIR, f)
            for f in os.listdir(TRAIN_IMAGES_DIR)
            if DistillationDataset._is_image_file(f)
        ])
        train_image_paths = random.sample(all_imgs, min(args.debug_max_train_images, len(all_imgs)))
        print(f"[DEBUG] Single-View: Limited train to {len(train_image_paths)} IMAGES")
    
    data_loader_train = build_distillation_dataloader(
        image_dir=TRAIN_IMAGES_DIR,
        features_dir=TRAIN_FEATURES_DIR if args.precomputed_features else None,
        teacher_extractor=teacher_extractor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        image_paths=train_image_paths, # Sarà None in multi-view mode
        distributed=args.distributed.distributed,
        multi_view_mode=args.multi_view_mode,
        split=TRAIN_SPLIT,             # "train": attiva Random Sampling delle view
        max_views_per_scene=args.max_views,
    )

    # Logica Debug per MULTI-VIEW: tagliamo la lista delle scene DOPO aver creato il dataset
    if args.multi_view_mode and args.debug_max_train_images:
        original_len = len(data_loader_train.dataset.samples)
        limit = min(args.debug_max_train_images, original_len)
        data_loader_train.dataset.samples = data_loader_train.dataset.samples[:limit]
        print(f"[DEBUG] Multi-View: Limited train to first {limit} SCENES (was {original_len})")

    
    # --- 2. VAL DATALOADER ---
    print(f"Building val dataloader from {VAL_IMAGES_DIR}")
    val_image_paths = None
    
    # Logica Debug per SINGLE-VIEW
    if args.debug_max_val_images and not args.multi_view_mode:
        all_val_imgs = sorted([
            os.path.join(VAL_IMAGES_DIR, f)
            for f in os.listdir(VAL_IMAGES_DIR)
            if DistillationDataset._is_image_file(f)
        ])
        val_image_paths = all_val_imgs[:args.debug_max_val_images]
        print(f"[DEBUG] Single-View: Limited val to {len(val_image_paths)} IMAGES")
    
    data_loader_val = build_distillation_dataloader(
        image_dir=VAL_IMAGES_DIR,
        features_dir=VAL_FEATURES_DIR if args.precomputed_features else None,
        teacher_extractor=teacher_extractor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        image_paths=val_image_paths, # Sarà None in multi-view mode
        distributed=args.distributed.distributed,
        multi_view_mode=args.multi_view_mode,
        split=VAL_SPLIT,               # "val": attiva Deterministic Slicing (prime N views)
        max_views_per_scene=args.max_views,
    )

    # Logica Debug per MULTI-VIEW
    if args.multi_view_mode and args.debug_max_val_images:
        original_len = len(data_loader_val.dataset.samples)
        limit = min(args.debug_max_val_images, original_len)
        data_loader_val.dataset.samples = data_loader_val.dataset.samples[:limit]
        print(f"[DEBUG] Multi-View: Limited val to first {limit} SCENES (was {original_len})")
    
    # ========== LOAD MODEL ==========
    print("Loading MapAnything model...")

    # Patch config file based on use_encoder_features
    if CONFIG_JSON_PATH and os.path.exists(CONFIG_JSON_PATH):
        import json
        with open(CONFIG_JSON_PATH, 'r') as f:
            config = json.load(f)
        
        # # DEBUG: Print original config
        # print("[DEBUG] Original config content:")
        # print(json.dumps(config, indent=2))
        
        # Modify input_feature_dims in feature_head_2 based on use_encoder_features
        if args.use_encoder_features:
            new_dims = [1024, 1024, 1024, 1024]
            print(f"[INFO] use_encoder_features=True: Setting feature_head_2.input_feature_dims to {new_dims}")
        else:
            new_dims = [1024, 768, 768, 768]
            print(f"[INFO] use_encoder_features=False: Setting feature_head_2.input_feature_dims to {new_dims}")
        
        if "pred_head_config" in config and "feature_head_2" in config["pred_head_config"]:
            config["pred_head_config"]["feature_head_2"]["input_feature_dims"] = new_dims
            
            # # DEBUG: Print modified config before save
            # print("[DEBUG] Modified config content (before save):")
            # print(json.dumps(config, indent=2))
            
            # Save modified config back to original location to overwrite
            with open(CONFIG_JSON_PATH, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"[INFO] Overwritten config file: {CONFIG_JSON_PATH}")
            
            # # DEBUG: Print config after save (read it back)
            # with open(CONFIG_JSON_PATH, 'r') as f:
            #     config_after = json.load(f)
            # print("[DEBUG] Config content (after save, read back):")
            # print(json.dumps(config_after, indent=2))

    # if global_rank == 0:
    #     model = MapAnything.from_pretrained(args.model_name, strict=False).to(device)
    # if torch.distributed.is_initialized():
    #     torch.distributed.barrier()
    # if global_rank != 0:
    #     model = MapAnything.from_pretrained(args.model_name, strict=False).to(device)

    if global_rank == 0:
        model = MapAnything.from_pretrained(
            args.model_name,
            revision="562de9ff7077addd5780415661c5fb031eb8003e",
            strict=False,
            # local_files_only=True,
        ).to(device)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if global_rank != 0:
        model = MapAnything.from_pretrained(
            args.model_name,
            revision="562de9ff7077addd5780415661c5fb031eb8003e",
            strict=False,
            # local_files_only=True,
        ).to(device)
    
    model_without_ddp = model
    print(f"Model loaded. Has dpt_feature_head_2: {hasattr(model, 'dpt_feature_head_2')}")

    # ========== OPTIONAL: INIT SAM2 DECODER TEACHER + ATTACH STUDENT DECODER ==========
    sam_prompt_encoder_teacher = None
    sam_mask_decoder_teacher = None
    if getattr(args, "distill_decoder", False):
        print(f"[INFO] Initializing SAM2 MaskDecoder teacher from {SAM2_PATH}")
        sam_prompt_encoder_teacher, sam_mask_decoder_teacher = load_sam2_teacher_prompt_and_decoder(
            checkpoint_path=SAM2_PATH,
            device=device,
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
        )
        print("[INFO] Attaching SAM2 MaskDecoder student (from scratch) to MapAnything model")
        # Student decoder matches SAM2 base hyperparams (see sam2_minimal.modeling.sam2_base.SAM2Base._build_sam_heads)
        model.sam2_mask_decoder_student = _build_sam_mask_decoder(embed_dim=256)

    # ========== FREEZE STRATEGY ==========
    # 1. Freeze tutto inizialmente
    print("Freezing all parameters...")
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. Unfreeze dpt_feature_head_2 e sam2_compat (sempre trainable)
    print("Unfreezing dpt_feature_head_2 and sam2_compat...")
    for name, param in model.named_parameters():
        if name.startswith("dpt_feature_head_2") or name.startswith("sam2_compat"):
            param.requires_grad = True

    # 2b. Unfreeze SAM2 mask decoder student (optional)
    if getattr(args, "distill_decoder", False) and hasattr(model, "sam2_mask_decoder_student"):
        print("Unfreezing sam2_mask_decoder_student...")
        for p in model.sam2_mask_decoder_student.parameters():
            p.requires_grad = True

    # 3. Unfreeze ultimi N blocchi di info_sharing.self_attention_blocks (opzionale)
    num_info_sharing_blocks = getattr(args, 'num_info_sharing_blocks_unfreeze', 0)
    if num_info_sharing_blocks > 0 and hasattr(model, "info_sharing"):
        info_sharing = model.info_sharing
        
        # Trova i blocchi (self_attention_blocks per MapAnything)
        if hasattr(info_sharing, "self_attention_blocks"):
            blocks = info_sharing.self_attention_blocks
        elif hasattr(info_sharing, "blocks"):
            blocks = info_sharing.blocks
        elif hasattr(info_sharing, "layers"):
            blocks = info_sharing.layers
        else:
            print("[WARN] info_sharing has no 'self_attention_blocks', 'blocks', or 'layers'. Skipping unfreezing.")
            blocks = []
        
        if len(blocks) > 0:
            start_idx = max(0, len(blocks) - num_info_sharing_blocks)
            unfrozen_count = 0
            unfrozen_indices = []
            for i in range(start_idx, len(blocks)):
                for param in blocks[i].parameters():
                    param.requires_grad = True
                    unfrozen_count += param.numel()
                unfrozen_indices.append(i)
            if num_info_sharing_blocks == 24:
                for name, p in model.named_parameters():
                    if name.startswith(("info_sharing.proj_embed", "info_sharing.norm")):
                        p.requires_grad = True
            args.info_sharing_unfrozen_indices = unfrozen_indices
            print(f"[INFO] Unfroze last {num_info_sharing_blocks} info_sharing blocks (indices {unfrozen_indices})")
            print(f"[INFO] Unfroze {unfrozen_count:,} parameters in info_sharing")
        else:
            args.info_sharing_unfrozen_indices = []
    else:
        args.info_sharing_unfrozen_indices = []

    # 4. Unfreeze ultimi N blocchi di DINOv2 encoder (opzionale)
    num_dino_layers_unfreeze = getattr(args, 'num_dino_layers_unfreeze', 0)
    if num_dino_layers_unfreeze > 0:
        # Trova l'encoder DINOv2
        dino_encoder = None
        if hasattr(model, "encoder"):
            dino_encoder = model.encoder
        elif hasattr(model, "dinov2_encoder"):
            dino_encoder = model.dinov2_encoder
        elif hasattr(model, "backbone"):
            dino_encoder = model.backbone
        
        if dino_encoder is not None:
            # Cerca i blocchi transformer dell'encoder
            blocks = None
            if hasattr(dino_encoder, "blocks"):
                blocks = dino_encoder.blocks
            elif hasattr(dino_encoder, "layers"):
                blocks = dino_encoder.layers
            elif hasattr(dino_encoder, "transformer"):
                if hasattr(dino_encoder.transformer, "blocks"):
                    blocks = dino_encoder.transformer.blocks
                elif hasattr(dino_encoder.transformer, "layers"):
                    blocks = dino_encoder.transformer.layers
            elif hasattr(dino_encoder, "model"):
                if hasattr(dino_encoder.model, "blocks"):
                    blocks = dino_encoder.model.blocks
                elif hasattr(dino_encoder.model, "layers"):
                    blocks = dino_encoder.model.layers
            
            if blocks is not None and len(blocks) > 0:
                start_idx = max(0, len(blocks) - num_dino_layers_unfreeze)
                unfrozen_count = 0
                unfrozen_dino_indices = []
                for i in range(start_idx, len(blocks)):
                    for param in blocks[i].parameters():
                        param.requires_grad = True
                        unfrozen_count += param.numel()
                    unfrozen_dino_indices.append(i)
                if num_dino_layers_unfreeze == 24:
                    for name, p in model.named_parameters():
                        if name.startswith((
                            "encoder.model.patch_embed.proj",
                            "encoder.model.pos_embed",
                            "encoder.model.cls_token",
                            "encoder.model.norm",
                        )):
                            p.requires_grad = True

                args.dino_unfrozen_indices = unfrozen_dino_indices
                print(f"[INFO] Unfroze last {num_dino_layers_unfreeze} DINOv2 encoder blocks (indices {unfrozen_dino_indices})")
                print(f"[INFO] Unfroze {unfrozen_count:,} parameters in DINOv2 encoder")
            else:
                print("[WARN] DINOv2 encoder has no 'blocks' or 'layers' attribute. Skipping unfreezing.")
                args.dino_unfrozen_indices = []
        else:
            print("[WARN] DINOv2 encoder not found on model. Skipping unfreezing.")
            args.dino_unfrozen_indices = []
    else:
        args.dino_unfrozen_indices = []

    # ========== DEBUG: Print DINOv2 encoder layer names ==========
    # dino_encoder = None
    # if hasattr(model, "encoder"):
    #     dino_encoder = model.encoder
    # elif hasattr(model, "dinov2_encoder"):
    #     dino_encoder = model.dinov2_encoder
    # elif hasattr(model, "backbone"):
    #     dino_encoder = model.backbone
    
    # if dino_encoder is not None:
    #     print("\n" + "="*80)
    #     print("DINOv2 ENCODER LAYER NAMES")
    #     print("="*80)
    #     for name, module in dino_encoder.named_modules():
    #         if name and ("block" in name.lower() or "layer" in name.lower()):
    #             print(f"  {name}")
    #     print("="*80 + "\n")
    # else:
    #     print("[WARN] DINOv2 encoder not found on model for debugging")

    # ========== VERIFY TRAINABLE PARAMETERS ==========
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = [p for p in model.parameters() if not p.requires_grad]
    
    trainable_count = sum(p.numel() for p in trainable_params)
    frozen_count = sum(p.numel() for p in frozen_params)
    total_count = trainable_count + frozen_count
    
    print("\n" + "="*80)
    print("TRAINABLE PARAMETERS SUMMARY")
    print("="*80)
    print(f"Trainable params: {trainable_count:,} ({100*trainable_count/total_count:.2f}%)")
    print(f"Frozen params:    {frozen_count:,} ({100*frozen_count/total_count:.2f}%)")
    print(f"Total params:     {total_count:,}")
    
    # Lista dei gruppi trainable
    trainable_groups = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Estrai il gruppo (es. "info_sharing.self_attention_blocks.10")
            parts = name.split(".")
            if len(parts) >= 4 and parts[0] == "info_sharing" and parts[1] == "self_attention_blocks":
                # info_sharing.self_attention_blocks.10.attn.qkv.weight -> info_sharing.self_attention_blocks.10
                group = ".".join(parts[:3])
            elif len(parts) >= 3:
                group = ".".join(parts[:3])
            else:
                group = ".".join(parts[:2])
            
            if group not in trainable_groups:
                trainable_groups[group] = 0
            trainable_groups[group] += param.numel()
    
    print("\n📦 Trainable parameter groups:")
    for group, count in sorted(trainable_groups.items()):
        print(f"   - {group}: {count:,} params")
    print("="*80 + "\n")

    log_param_status(model, max_print=None)
    
    # Initialize criterion
    criterion = DistillationLoss(
        mse_weight=args.mse_weight,
        cosine_weight=args.cosine_weight,
        normalize=args.normalize_features,
    ).to(device)

    # ========== OPTIMIZER con LR differenziati ==========
    head_params = []
    transformer_params = []
    encoder_params = []
    decoder_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("dpt_feature_head_2") or name.startswith("sam2_compat"):
            head_params.append(p)
        elif name.startswith("sam2_mask_decoder_student"):
            decoder_params.append(p)
        elif name.startswith("info_sharing"):
            transformer_params.append(p)
        elif name.startswith("encoder") and hasattr(args, 'dino_unfrozen_indices') and args.dino_unfrozen_indices:
            encoder_params.append(p)
        else:
            other_params.append(p)

    # Fallback: se alcuni parametri trainabili non rientrano nelle categorie, mettili nel gruppo head
    if other_params:
        print(f"[WARN] {sum(op.numel() for op in other_params):,} trainable params not matched; assigning to HEAD LR group.")
        head_params.extend(other_params)

    lr_head = args.lr
    lr_encoder = args.lr * args.lr_encoder_scale
    lr_transformer = args.lr * args.lr_transformer_scale
    lr_decoder = getattr(args, "lr_decoder", None)
    if lr_decoder is None:
        lr_decoder = args.lr * float(getattr(args, "lr_decoder_scale", 1.0))

    optimizer = optim.AdamW(
        [
            {"params": head_params, "lr": lr_head},
            {"params": transformer_params, "lr": lr_transformer},
            {"params": encoder_params, "lr": lr_encoder},
            {"params": decoder_params, "lr": lr_decoder},
        ],
        lr=args.lr,  # non usato per i gruppi espliciti, rimane come default
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    print(f"[OPT] Groups: head={sum(p.numel() for p in head_params):,} params @ LR {lr_head}, "
          f"encoder={sum(p.numel() for p in encoder_params):,} params @ LR {lr_encoder}, "
          f"transformer={sum(p.numel() for p in transformer_params):,} params @ LR {lr_transformer}, "
          f"decoder={sum(p.numel() for p in decoder_params):,} params @ LR {lr_decoder}")
    
    # ========== WRAPPING IN DDP ==========
    if args.distributed.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.distributed.gpu],
            find_unused_parameters=False,
        )
        model_without_ddp = model.module
        
        # If the module graph is static across iterations, avoid re-registering DDP hooks every iteration.
        # This prevents errors like "marked ready twice" when using checkpointing / reentrant autograd.
        try:
            if hasattr(model, "_set_static_graph"):
                model._set_static_graph()
        except Exception:
            pass
    
    # Scheduler LR: Cosine annealing per epoca, coerente con distillation.py
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.lr_scheduler_t_max,
            eta_min=args.lr_min,
        )
        print(f"[INFO] Using CosineAnnealingLR with T_max={args.lr_scheduler_t_max}, eta_min={args.lr_min}")
    elif args.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_decay_epochs,
            gamma=0.1,
        )
        print(f"[INFO] Using StepLR with step_size={args.lr_decay_epochs}, gamma=0.1")
    elif args.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=args.plateau_patience,
            threshold=1e-3,
            threshold_mode='abs',
            cooldown=1,
            min_lr=args.lr_min,
        )
        print(f"[INFO] Using ReduceLROnPlateau (factor=0.5, patience={args.plateau_patience}, threshold=1e-3 abs, cooldown=1)")
    else:
        print(f"[INFO] Learning rate scheduler disabled. LR will remain constant at {args.lr}")
    
    # Resume: ricarica head 2 + optimizer + scheduler; riparte dall'epoca successiva
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume_ckpt:
        print(f"Resuming from checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        model_without_ddp.dpt_feature_head_2.load_state_dict(ckpt["dpt_feature_head_2"])

        # Load sam2_compat if present in checkpoint
        if "sam2_compat" in ckpt and hasattr(model_without_ddp, "sam2_compat"):
            model_without_ddp.sam2_compat.load_state_dict(ckpt["sam2_compat"])
            print("[INFO] Loaded sam2_compat state from checkpoint")
        elif hasattr(model_without_ddp, "sam2_compat"):
            print("[WARN] sam2_compat exists on model but not found in checkpoint. Using random initialization.")

        # Load SAM2 mask decoder student if present
        if getattr(args, "distill_decoder", False) and hasattr(model_without_ddp, "sam2_mask_decoder_student"):
            if "sam2_mask_decoder_student" in ckpt:
                model_without_ddp.sam2_mask_decoder_student.load_state_dict(ckpt["sam2_mask_decoder_student"], strict=True)
                print("[INFO] Loaded sam2_mask_decoder_student state from checkpoint")
            else:
                print("[WARN] distill_decoder=True but sam2_mask_decoder_student not found in checkpoint. Using random init.")

        # Restore unfrozen info_sharing blocks if present
        if "info_sharing_blocks" in ckpt and hasattr(model_without_ddp, "info_sharing"):
            info = model_without_ddp.info_sharing
            if hasattr(info, "self_attention_blocks"):
                blocks = info.self_attention_blocks
            elif hasattr(info, "blocks"):
                blocks = info.blocks
            elif hasattr(info, "layers"):
                blocks = info.layers
            else:
                blocks = []
            saved_indices = ckpt.get("info_sharing_unfrozen_indices", [])
            missing = []
            for idx, state_dict in ckpt["info_sharing_blocks"].items():
                if idx < len(blocks):
                    try:
                        blocks[idx].load_state_dict(state_dict)
                    except Exception as e:
                        print(f"[WARN] Failed loading info_sharing block {idx}: {e}")
                else:
                    missing.append(idx)
            print(f"[INFO] Restored unfrozen info_sharing blocks from checkpoint: {saved_indices}")
            if missing:
                print(f"[WARN] Saved block indices not present in current model: {missing}")
            args.info_sharing_unfrozen_indices = saved_indices

            if "info_sharing_wrappers" in ckpt:
                for name, data in ckpt["info_sharing_wrappers"].items():
                    param = dict(model_without_ddp.info_sharing.named_parameters())[name]
                    param.data.copy_(data)
                    param.requires_grad = True
                print(f"[RESUME] Restored {len(ckpt['info_sharing_wrappers'])} info_sharing wrapper params")

        # Restore unfrozen DINOv2 blocks
        if "dino_encoder_blocks" in ckpt and hasattr(model_without_ddp, "encoder"):
            dino_encoder = model_without_ddp.encoder
            if hasattr(dino_encoder, "blocks"):
                blocks = dino_encoder.blocks
            elif hasattr(dino_encoder, "transformer") and hasattr(dino_encoder.transformer, "blocks"):
                blocks = dino_encoder.transformer.blocks
            elif hasattr(dino_encoder, "model"):
                if hasattr(dino_encoder.model, "blocks"):
                    blocks = dino_encoder.model.blocks
                elif hasattr(dino_encoder.model, "layers"):
                    blocks = dino_encoder.model.layers
            else:
                blocks = []
            
            for idx, state_dict in ckpt["dino_encoder_blocks"].items():
                if idx < len(blocks):
                    try:
                        blocks[idx].load_state_dict(state_dict)
                    except Exception as e:
                        print(f"[WARN] Failed loading DINOv2 block {idx}: {e}")
            print(f"[INFO] Restored unfrozen DINOv2 encoder blocks from checkpoint")
            args.dino_unfrozen_indices = list(ckpt["dino_encoder_blocks"].keys())

            if "dino_encoder_wrappers" in ckpt:
                dino_model = dino_encoder.model if hasattr(dino_encoder, "model") else dino_encoder
                for name, data in ckpt["dino_encoder_wrappers"].items():
                    param = dict(dino_model.named_parameters())[name]
                    param.data.copy_(data)
                    param.requires_grad = True
                print(f"[RESUME] Restored {len(ckpt['dino_encoder_wrappers'])} DINOv2 wrapper params")

        optimizer.load_state_dict(ckpt["optimizer"])

        if args.lr_scheduler == "none" or args.override_lr:
            # Rispetta i gruppi differenziati: head, transformer, encoder
            optimizer.param_groups[0]['lr'] = args.lr                          # head
            optimizer.param_groups[1]['lr'] = args.lr * args.lr_transformer_scale  # transformer
            optimizer.param_groups[2]['lr'] = args.lr * args.lr_encoder_scale      # encoder
            print(f"[INFO] Overriding optimizer LR: head={optimizer.param_groups[0]['lr']}, "
                f"transformer={optimizer.param_groups[1]['lr']}, encoder={optimizer.param_groups[2]['lr']}")

        # Scheduler resume logic
        if args.lr_scheduler != "none" and "scheduler" in ckpt and not args.override_scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
            # If user provided a new T_max, overwrite it in the scheduler
            if hasattr(scheduler, "T_max") and getattr(args, "overwrite_scheduler_t_max", False):
                old_tmax = getattr(scheduler, "T_max", None)
                scheduler.T_max = args.lr_scheduler_t_max
                print(f"[INFO] Overriding scheduler T_max: {old_tmax} -> {scheduler.T_max}")
        elif args.override_scheduler or args.lr_scheduler != "none":
            print(f"[INFO] Using NEW scheduler from CLI: {args.lr_scheduler} (not loading from checkpoint)")

            # [NUOVO] Se override scheduler, avanza manualmente per recuperare gli step persi
            if args.override_scheduler and scheduler is not None:
                resumed_epoch = ckpt.get("epoch", 0) + 1
                
                if args.lr_scheduler == "step":
                    # StepLR: Chiama step() per ogni epoca già completata
                    for _ in range(resumed_epoch):
                        scheduler.step()
                    print(f"[INFO] Advanced StepLR scheduler by {resumed_epoch} steps (current LR: {optimizer.param_groups[0]['lr']:.6e})")
                
                elif args.lr_scheduler == "cosine":
                    # CosineAnnealingLR: Salta direttamente all'epoca corrente
                    scheduler.last_epoch = resumed_epoch - 1  # last_epoch è 0-indexed
                    scheduler.step()  # Aggiorna LR basato su last_epoch
                    print(f"[INFO] Set CosineAnnealingLR to epoch {resumed_epoch} (current LR: {optimizer.param_groups[0]['lr']:.6e})")

        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
    
    # Inizializzazione opzionale di W&B (solo rank 0): salva anche config e gestisce resume
    if args.use_wandb and WANDB_AVAILABLE and global_rank == 0:
        wandb_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_name,
            "config": vars(args),
        }
        if args.wandb_resume_id:
            wandb_kwargs.update(id=args.wandb_resume_id, resume="allow")
        wandb.init(**wandb_kwargs)
    
    # ========== TRAINING LOOP ==========
    print(f"Start distillation training for {args.epochs} epochs from epoch {start_epoch}")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        if args.distributed.distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
        
        epoch_start = time.time()
        
        # Train one epoch (passa teacher_extractor)
        train_stats = train_one_epoch_distillation(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            teacher_extractor=teacher_extractor,
            sam_prompt_encoder_teacher=sam_prompt_encoder_teacher,
            sam_mask_decoder_teacher=sam_mask_decoder_teacher,
        )
        
        # Validation
        val_stats = {}
        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
            val_stats = validate_one_epoch_distillation(
                model=model,
                criterion=criterion,
                data_loader=data_loader_val,
                device=device,
                epoch=epoch,
                args=args,
                teacher_extractor=teacher_extractor,
                sam_prompt_encoder_teacher=sam_prompt_encoder_teacher,
                sam_mask_decoder_teacher=sam_mask_decoder_teacher,
            )
            
            # Check for new best
            val_loss_avg = val_stats.get("loss_avg", float("inf"))
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                print(f"New best validation loss: {best_val_loss:.6f}")
                # Save best checkpoint
                if global_rank == 0:
                    save_checkpoint_distillation(
                        model_without_ddp,
                        optimizer,
                        scheduler,
                        epoch,
                        best_val_loss,
                        args.output_dir,
                        tag="best",
                        args=args,
                    )
        
        # Step scheduler
        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                # usa la loss di validazione come metrica
                if val_stats:  # step una volta per epoca, dopo la val
                    scheduler.step(val_stats.get("loss_avg", float("inf")))
            else:
                scheduler.step()
            # Optional: stop here to inspect scheduler/optimizer state via pdb
            if getattr(args, "debug_pdb_lr", False):
                import pdb
                pdb.set_trace()
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            if global_rank == 0:
                save_checkpoint_distillation(
                    model_without_ddp,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_loss,
                    args.output_dir,
                    tag=f"epoch{epoch+1}",
                    args=args,
                )
        
        epoch_time = time.time() - epoch_start

        # Log to wandb (match distillation.py keys)
        if args.use_wandb and WANDB_AVAILABLE and global_rank == 0:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_stats.get("loss_mean", 0.0),
                "train_mse_loss": train_stats.get("mse_loss_mean", 0.0),
                "train_cos_loss": train_stats.get("cos_loss_mean", 0.0),
                "train_mean_diff": train_stats.get("mean_diff", 0.0),
                "train_std_diff": train_stats.get("std_diff", 0.0),
                "train_cos_sim": train_stats.get("cos_sim_mean", 0.0),
                "train_decoder_loss": train_stats.get("dec_loss_mean", 0.0),
                "train_dec_mse_masks": train_stats.get("dec_mse_masks_mean", 0.0),
                "train_dec_mse_iou": train_stats.get("dec_mse_iou_mean", 0.0),
                "train_dec_mse_tokens": train_stats.get("dec_mse_tokens_mean", 0.0),
                "train_dec_mse_obj": train_stats.get("dec_mse_obj_mean", 0.0),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_sec": epoch_time,
            }
            if val_stats:
                print(f"[DEBUG] Logging validation stats to W&B: {val_stats}")  # <--- DEBUG
                log_dict.update({
                    "val_loss": val_stats.get("loss_mean", 0.0),
                    "val_mse_loss": val_stats.get("mse_loss_mean", 0.0),
                    "val_cos_loss": val_stats.get("cos_loss_mean", 0.0),
                    "val_mean_diff": val_stats.get("mean_diff", 0.0),
                    "val_std_diff": val_stats.get("std_diff", 0.0),
                    "val_cosine_similarity": val_stats.get("cos_sim_mean", 0.0),
                    "val_decoder_loss": val_stats.get("dec_loss_mean", 0.0),
                    "val_dec_mse_masks": val_stats.get("dec_mse_masks_mean", 0.0),
                    "val_dec_mse_iou": val_stats.get("dec_mse_iou_mean", 0.0),
                    "val_dec_mse_tokens": val_stats.get("dec_mse_tokens_mean", 0.0),
                    "val_dec_mse_obj": val_stats.get("dec_mse_obj_mean", 0.0),
                })
            else:
                print(f"[DEBUG] val_stats is empty, skipping validation logging for epoch {epoch+1}")  # <--- DEBUG
            wandb.log(log_dict)
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_stats.get('loss_mean', 0):.6f} | "
            f"Val Loss: {val_stats.get('loss_mean', 0):.6f} | "
            f"Time: {epoch_time:.2f}s"
        )
    
    # Save final checkpoint
    if global_rank == 0:
        save_checkpoint_distillation(
            model_without_ddp,
            optimizer,
            scheduler,
            args.epochs - 1,
            best_val_loss,
            args.output_dir,
            tag="final",
            args=args,
        )
    
    total_time = time.time() - start_time
    print(f"Distillation training completed in {str(datetime.timedelta(seconds=int(total_time)))}")
    
    if args.use_wandb and WANDB_AVAILABLE and global_rank == 0:
        wandb.finish()

# ==================== Checkpoint Management ====================
def save_checkpoint_distillation(
    model_without_ddp,
    optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    output_dir: str,
    tag: str = "last",
    args=None,
):
    """
    Save checkpoint containing only dpt_feature_head_2 and optimizer state.
    
    Args:
        model_without_ddp: Model without DDP wrapper
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        best_val_loss: Best validation loss so far
        output_dir: Directory to save checkpoint
        tag: Tag for checkpoint filename (e.g., "best", "last", "epoch10")
    """
    state = {
        "dpt_feature_head_2": model_without_ddp.dpt_feature_head_2.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }

    # Save sam2_compat if present
    if hasattr(model_without_ddp, "sam2_compat"):
        state["sam2_compat"] = model_without_ddp.sam2_compat.state_dict()
        # print("[INFO] sam2_compat state added to checkpoint")

    # Save SAM2 mask decoder student if present
    if hasattr(model_without_ddp, "sam2_mask_decoder_student"):
        state["sam2_mask_decoder_student"] = model_without_ddp.sam2_mask_decoder_student.state_dict()

    # Save unfrozen info_sharing blocks if any
    if args is not None and hasattr(model_without_ddp, "info_sharing") and getattr(args, "info_sharing_unfrozen_indices", []):
        info = model_without_ddp.info_sharing
        if hasattr(info, "self_attention_blocks"):
            blocks = info.self_attention_blocks
        elif hasattr(info, "blocks"):
            blocks = info.blocks
        elif hasattr(info, "layers"):
            blocks = info.layers
        else:
            blocks = []
        indices = [i for i in getattr(args, "info_sharing_unfrozen_indices", []) if i < len(blocks)]
        state["info_sharing_unfrozen_indices"] = indices
        state["info_sharing_blocks"] = {i: blocks[i].state_dict() for i in indices}
        print(f"[INFO] Added {len(indices)} unfrozen info_sharing blocks to checkpoint: {indices}")

        if args.num_info_sharing_blocks_unfreeze == 24:
            wrapper_state = {}
            for name, param in model_without_ddp.info_sharing.named_parameters():
                if name.startswith(("proj_embed", "norm")):
                    wrapper_state[name] = param.data.clone()
            if wrapper_state:
                state["info_sharing_wrappers"] = wrapper_state
                print(f"[INFO] Added {len(wrapper_state)} info_sharing wrapper params to checkpoint")

    # Save unfrozen DINOv2 blocks
    if args is not None and hasattr(args, 'dino_unfrozen_indices') and args.dino_unfrozen_indices:
        dino_encoder = model_without_ddp.encoder
        if hasattr(dino_encoder, "blocks"):
            blocks = dino_encoder.blocks
        elif hasattr(dino_encoder, "transformer") and hasattr(dino_encoder.transformer, "blocks"):
            blocks = dino_encoder.transformer.blocks
        elif hasattr(dino_encoder, "model"):
            if hasattr(dino_encoder.model, "blocks"):
                blocks = dino_encoder.model.blocks
            elif hasattr(dino_encoder.model, "layers"):
                blocks = dino_encoder.model.layers
        else:
            blocks = []
        
        if blocks:
            state["dino_encoder_blocks"] = {}
            for idx in args.dino_unfrozen_indices:
                if idx < len(blocks):
                    state["dino_encoder_blocks"][idx] = blocks[idx].state_dict()

        if args.num_dino_layers_unfreeze == 24:
            wrapper_state = {}
            # Trova il modello DINO effettivo (potrebbe essere sotto .model)
            dino_model = dino_encoder.model if hasattr(dino_encoder, "model") else dino_encoder
            for name, param in dino_model.named_parameters():
                if name.startswith(("patch_embed.proj", "pos_embed", "cls_token", "norm")):
                    wrapper_state[name] = param.data.clone()
            if wrapper_state:
                state["dino_encoder_wrappers"] = wrapper_state
                print(f"[INFO] Added {len(wrapper_state)} DINOv2 wrapper params to checkpoint")

        print(f"[INFO] Added {len(state['dino_encoder_blocks'])} unfrozen DINOv2 encoder blocks to checkpoint: {list(state['dino_encoder_blocks'].keys())}")

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    
    # Save wandb run_id if available
    if WANDB_AVAILABLE and wandb.run is not None:
        state["wandb_run_id"] = wandb.run.id
    
    # Crea la sottocartella checkpoints
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = ckpt_dir / f"checkpoint_{tag}.pth"  # Ora salva in checkpoints/
    torch.save(state, ckpt_path)
    print(f"[SAVE] Checkpoint saved: {ckpt_path}")

# ==================== Argument Parser ====================
def get_args_parser():
    """
    Create argument parser for distillation training.
    
    Returns:
        argparse.ArgumentParser
    """
    # Parser degli argomenti CLI: organizza le opzioni per eseguire il training di distillazione
    parser = argparse.ArgumentParser(
        description="Distillation training for MapAnything with SAM2 encoder features"
    )
    
    # Paths
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints and logs (default: OUT_DIR/wandb_name or timestamp)")
    # Note: dataset paths are derived from COCO2017_ROOT constants depending on run_cluster
    
    # Model
    # Config file path: /scratch/.cache/niacobone/huggingface/hub/models--facebook--map-anything/snapshots/6f3a25bfbb8fcc799176bb01e9d07dfb49d5416a/config.json
    parser.add_argument("--model_name", type=str, default="facebook/map-anything", help="MapAnything model name or path")
    # parser.add_argument("--model_revision", type=str, default="6f3a25bfbb8fcc799176bb01e9d07dfb49d5416a", help="HF snapshot hash to pin")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--accum_iter", type=int, default=1, help="Gradient accumulation iterations")
    parser.add_argument("--use_encoder_features", action="store_true", help="Use encoder features instead of transformer features for distillation")
    parser.add_argument("--debug_pdb_lr", action="store_true", help="Stop after optimizer.step() to inspect LRs via pdb")
    
    # Learning rate and scheduler
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate") # Usa 5e-4 per BS_eff = 16 --> 1e-3 per BS_eff = 32, 2.5e-4 per BS_eff = 8
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum learning rate for scheduler")
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=["cosine","step", "plateau", "none"])
    parser.add_argument("--plateau_patience", type=int, default=10, help="Patience for ReduceLROnPlateau scheduler")
    parser.add_argument("--lr_decay_epochs", type=int, default=1000, help="Epochs per decay x0.1 (StepLR)")
    parser.add_argument("--lr_scheduler_t_max", type=int, default=None, help="T_max for CosineAnnealingLR")
    parser.add_argument("--override_lr", action="store_true", help="Override LR from checkpoint with --lr value")
    parser.add_argument("--overwrite_scheduler_t_max", action="store_true", help="Overwrite scheduler T_max when resuming")
    parser.add_argument("--override_scheduler", action="store_true", help="Override scheduler from checkpoint with CLI args")
    
    # Mixed precision
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="AMP dtype")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--disable_cudnn_benchmark", action="store_true", help="Disable cudnn benchmark")

    # Loss
    parser.add_argument("--mse_weight", type=float, default=0.5, help="Weight for MSE loss")
    parser.add_argument("--cosine_weight", type=float, default=0.5, help="Weight for cosine loss")
    parser.add_argument("--mse_type", type=str, default="sample", choices=["pixel", "sample"], help="Type of MSE loss computation")
    parser.add_argument("--normalize_features", action="store_true", help="Normalize features before loss")
    
    # Data
    parser.add_argument("--dataset", type=str, default="coco2017", choices=["coco2017", "ETH3D", "ETH3D_single"], help="Seleziona il dataset")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--debug_max_train_images", type=int, default=None, help="Limit training images for debugging")
    parser.add_argument("--debug_max_val_images", type=int, default=None, help="Limit validation images for debugging")
    parser.add_argument("--precomputed_features", action="store_true", help="Use precomputed features from disk")
    parser.add_argument("--no_augmentation", action="store_true", help="Disable data augmentation in online mode")
    
    # Multi-view
    parser.add_argument("--multi_view_mode", action="store_true", help="Enable multi-view mode (cross-attention between views)")
    parser.add_argument("--max_views", type=int, default=6, help="Max views per scene. Train: Random Sample. Val: First N.")
    
    # Checkpointing
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save_freq", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_freq", type=int, default=1, help="Run validation every N epochs")
    
    # Logging
    parser.add_argument("--print_freq", type=int, default=10, help="Print frequency (iterations)")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="mapanything-distillation", help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_resume_id", type=str, default=None, help="W&B run ID to resume")
    parser.add_argument("--save_visualizations", action="store_true", help="Save PCA visualizations during validation")
    parser.add_argument("--log_freq", type=int, default=100, help="Log to W&B every N batches")
    
    # Distributed (opzionale): abilita DDP; dist_url di solito 'env://' con torchrun; local_rank impostato da torchrun
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--dist_url", type=str, default="env://", help="URL for distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")

    # Unfreeze strategy
    parser.add_argument("--num_info_sharing_blocks_unfreeze", type=int, default=0, help="Number of last info_sharing transformer blocks to unfreeze") # max 24
    parser.add_argument("--num_dino_layers_unfreeze", type=int, default=0, help="Number of last DINOv2 encoder layers to unfreeze") # max 24
    parser.add_argument("--lr_encoder_scale", type=float, default=0.1, help="Scale factor for encoder LR relative to --lr")
    parser.add_argument("--lr_transformer_scale", type=float, default=1.0, help="Scale factor for transformer LR relative to --lr")

    # SAM2 decoder distillation
    parser.add_argument("--distill_decoder", action="store_true", help="Enable SAM2 MaskDecoder distillation (teacher SAM2 -> student from scratch)")
    parser.add_argument("--decoder_loss_weight", type=float, default=1.0, help="Weight for decoder distillation loss added to total loss")
    parser.add_argument("--lr_decoder", type=float, default=None, help="Absolute LR for decoder (overrides --lr_decoder_scale)")
    parser.add_argument("--lr_decoder_scale", type=float, default=1.0, help="Scale factor for decoder LR relative to --lr")
    parser.add_argument(
        "--decoder_multimask_output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, distill multimask outputs (3 masks). If false, distill single-mask output.",
    )
    
    # comando debug pc lab
    # python distillation_test_multi_view_gemini.py --epochs 5 --log_freq 1 --debug_max_train_images 10 --debug_max_val_images 5 --save_freq 1 --save_visualizations --num_info_sharing_blocks_unfreeze 2
    # python distillation_test_multi_view_gemini.py --epochs 10 --log_freq 1 --debug_max_train_images 10 --debug_max_val_images 5 --save_freq 1 --save_visualizations --num_info_sharing_blocks_unfreeze 4 --resume_ckpt /scratch2/nico/distillation/output/distill_20251125_143157/checkpoints/checkpoint_best.pth
    # python distillation_test_multi_view_gemini.py --epochs 10 --log_freq 1 --save_freq 1 --save_visualizations --multi_view_mode

    # Proporzioni
    # batch_size 1, lr 1e-4, accum_iter 1
    # batch_size 2, lr 2e-4, accum_iter 1
    # batch_size 1, lr 1e-4, accum_iter 2

    return parser

# ==================== Entry Point ====================
def main():
    """
    Entry point for distillation training script.
    """
    parser = get_args_parser()
    args = parser.parse_args()
    if args.lr_scheduler_t_max is None and args.lr_scheduler == "cosine":
        args.lr_scheduler_t_max = args.epochs  # Default T_max to epochs if not set
    
    # Crea un oggetto Namespace compatibile con train_tools
    # (train_tools si aspetta args.distributed come oggetto con attributi: distributed/dist_url/gpu).
    # Nota: in uno scenario torchrun si potrebbero leggere anche gli env (LOCAL_RANK, RANK, WORLD_SIZE)
    # per riempire automaticamente questi campi.
    if not hasattr(args, "distributed") or not isinstance(args.distributed, argparse.Namespace):
        distributed_args = argparse.Namespace(
            distributed=args.distributed if hasattr(args, "distributed") else False,
            dist_url=args.dist_url if hasattr(args, "dist_url") else "env://",
            gpu=args.local_rank if hasattr(args, "local_rank") else 0,
        )
        args.distributed = distributed_args
    
    # Run distillation
    try:
        distill(args)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user.")
    except Exception as e:
        print(f"[ERROR] Unexpected exception: {e}")
        raise
    finally:
        # Cleanup
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

if __name__ == "__main__":
    main()
