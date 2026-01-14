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
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image

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
from nico.utils import *

from sam2_minimal.modeling.sam.prompt_encoder import PromptEncoder
from sam2_minimal.modeling.sam.mask_decoder import MaskDecoder
from sam2_builder import (
    load_sam2_feature_extractor,
    load_sam2_teacher_prompt_and_decoder,
    build_sam_mask_decoder,
)

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
    global IMAGES_DIRNAME
    global TRAIN_IMAGES_DIR, VAL_IMAGES_DIR
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
    else:
        OUT_DIR = "/scratch2/nico/distillation/output"
        BASE_DIR = "/scratch2/nico/distillation/dataset"
        SAM2_PATH = "/scratch2/nico/sam2/checkpoints/sam2.1_hiera_large.pt"
        CONFIG_JSON_PATH = "/scratch/.cache/niacobone/huggingface/hub/models--facebook--map-anything/snapshots/562de9ff7077addd5780415661c5fb031eb8003e"

    # Usa args.dataset
    DATASET = args.dataset  # "coco2017" o "ETH3D"

    if DATASET == "coco2017":
        TRAIN_SPLIT = "train2017"
        VAL_SPLIT = "val2017"
    else:
        TRAIN_SPLIT = "train"
        VAL_SPLIT = "val"

    IMAGES_DIRNAME = "images"

    BASE_DIR = BASE_DIR + f"/{DATASET}"
    TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, IMAGES_DIRNAME, TRAIN_SPLIT)
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, IMAGES_DIRNAME, VAL_SPLIT)
    CONFIG_JSON_PATH = os.path.join(CONFIG_JSON_PATH, "config.json")

    print(f"[INFO] Using TRAIN_IMAGES_DIR: {TRAIN_IMAGES_DIR}")
    print(f"[INFO] Using VAL_IMAGES_DIR: {VAL_IMAGES_DIR}")

# ==================== Dataset Classes ====================
class DistillationDataset(Dataset):
    """
    Dataset per la distillazione: supporta Single-View (lista piatta) e Multi-View (scene folders).
    """
    
    def __init__(
        self,
        image_dir: str,
        teacher_extractor: Optional[callable] = None,
        image_paths: Optional[List[str]] = None,
        transform=None,
        multi_view_mode: bool = False,
        max_views_per_scene: int = 6,
        split: str = "train",
    ):
        self.image_dir = Path(image_dir)
        self.teacher_extractor = teacher_extractor
        self.transform = transform
        self.multi_view_mode = multi_view_mode
        self.max_views_per_scene = max_views_per_scene
        self.is_train = "train" in split.lower()
        
        # Validation
        if self.teacher_extractor is None:
            raise ValueError("Teacher_extractor must be provided")
        
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
                print(f"[INFO] Mode: MULTI-VIEW (Scenes) | Split: {split} | Max Views: {self.max_views_per_scene}")
            else:
                # --- LOGICA SINGLE-VIEW ---
                # Ogni "sample" è una stringa (path immagine)
                self.samples = sorted([
                    str(self.image_dir / f)
                    for f in os.listdir(self.image_dir)
                    if self._is_image_file(f)
                ])
                print(f"[INFO] Mode: SINGLE-VIEW (Images)")
                print(f"[INFO] Found {len(self.samples)} images in {image_dir}")

        print(f"[INFO] Using online feature extraction with teacher model")
    
    @staticmethod
    def _is_image_file(name: str) -> bool:
        return name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    
    def __len__(self) -> int:
        return len(self.samples)

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
            # Online mode: carica immagini PIL
            pil_images_list = []

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
        # extractor è un'istanza di SAM2FeatureExtractor che è a sua volta un wrapper che contiene image_encoder (trunk + neck)
        self.extractor = load_sam2_feature_extractor(checkpoint_path, device) 
        self.device = device
        self.augment_cfg = augment_cfg or {}
        self._build_augment_pipelines()
        print(f"[INFO] Loaded SAM2 feature extractor on {device}")

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
            loss_details: dizionario con componenti ('enc_mse_loss','enc_cos_loss','enc_cos_sim')
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
            "enc_mse_loss": mse_loss.item(),
            "enc_cos_loss": cos_loss.item(),
            "enc_cos_sim": cos_sim.item(),
        }
        
        return total_loss, loss_details

class DecoderDistillationLoss(torch.nn.Module):
    """
    Loss MSE per distillazione decoder SAM2.
    Applica MSE su masks, IoU predictions, e output tokens.
    """
    
    def __init__(
        self,
        weight_masks: float = 0.5,
        weight_iou: float = 0.3,
        weight_tokens: float = 1.0,
    ):
        super().__init__()
        self.weight_masks = weight_masks
        self.weight_iou = weight_iou
        self.weight_tokens = weight_tokens
    
    def forward(
        self,
        student_masks: torch.Tensor,
        student_iou: torch.Tensor,
        student_tokens: torch.Tensor,
        teacher_masks: torch.Tensor,
        teacher_iou: torch.Tensor,
        teacher_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Calcola loss MSE di distillazione decoder.
        
        Returns:
            total_loss: Loss scalare totale
            loss_dict: Dizionario con componenti individuali
        """
        # MSE per ogni componente
        loss_masks = F.mse_loss(student_masks, teacher_masks)
        loss_iou = F.mse_loss(student_iou, teacher_iou)
        loss_tokens = F.mse_loss(student_tokens, teacher_tokens)
        
        # Combinazione pesata
        total_loss = (
            self.weight_masks * loss_masks +
            self.weight_iou * loss_iou +
            self.weight_tokens * loss_tokens
        )
        
        loss_dict = {
            "decoder_loss_masks": loss_masks.item(),
            "decoder_loss_iou": loss_iou.item(),
            "decoder_loss_tokens": loss_tokens.item(),
            "decoder_loss_total": total_loss.item(),
        }
        
        return total_loss, loss_dict

# ==================== Data Loaders ====================
def build_distillation_dataloader(
    image_dir: str,
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
        teacher_extractor: TeacherFeatureExtractor instance
        batch_size: Batch size per GPU
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the dataset (ignored if distributed=True)
        image_paths: Optional list of specific image paths to use
        pin_memory: Whether to use pinned memory
        distributed: Whether to use DistributedSampler for multi-GPU training
    
    Returns:
        DataLoader per distillazione
    """
    dataset = DistillationDataset(
        image_dir=image_dir,
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

        if student_features is None:
            raise KeyError("Student features not found (_last_feat2_8x)")
        
        return student_features  # (B, C, H, W) dove B = len(image_paths)
    
def _generate_point_prompts_grid(batch_size: int, image_size: int = 1024, points_per_side: int = 3, device: torch.device = torch.device("cuda")) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Genera una griglia di punti prompts per il decoder (stile Automatic Mask Generator).
    
    Args:
        batch_size: Numero di immagini nel batch
        image_size: Dimensione dell'immagine (default 1024)
        points_per_side: Numero di punti per lato della griglia (default 3 -> 9 punti totali)
        device: Device CUDA
    
    Returns:
        point_coords: Tensor (B, K, 2) con coordinate (x, y) normalizzate
        point_labels: Tensor (B, K) con label (1 = positive prompt)
    """
    # Crea griglia di punti
    step = image_size // (points_per_side + 1)
    points_list = []
    
    for i in range(1, points_per_side + 1):
        for j in range(1, points_per_side + 1):
            points_list.append([step * i, step * j])
    
    K = len(points_list)  # Numero di punti
    
    # Replica per batch
    point_coords = torch.tensor(points_list, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # (B, K, 2)
    point_labels = torch.ones((batch_size, K), dtype=torch.int32, device=device)  # (B, K)
    
    return point_coords, point_labels

def train_one_epoch_distillation(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    decoder_criterion: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    teacher_extractor: Optional[TeacherFeatureExtractor] = None,
    sam_prompt_encoder_teacher=None,
    sam_mask_decoder_teacher=None,
    sam_mask_decoder_student=None,
) -> Dict:
    """
    Train the model for one epoch on distillation task.
    
    Args:
        model: MapAnything model with dpt_feature_head_2
        criterion: DistillationLoss instance for encoder
        decoder_criterion: DecoderDistillationLoss instance for decoder
        data_loader: DataLoader providing image paths and teacher features
        optimizer: Optimizer
        device: Device to run on
        epoch: Current epoch number
        args: Configuration namespace
        teacher_extractor: TeacherFeatureExtractor instance
        sam_prompt_encoder_teacher: Teacher prompt encoder (frozen)
        sam_mask_decoder_teacher: Teacher mask decoder (frozen)
        sam_mask_decoder_student: Student mask decoder (trainable)
    
    Returns:
        Dictionary of averaged training metrics
    """
    model.train(True)
    metric_logger = train_tools.MetricLogger(delimiter=" | ")
    metric_logger.add_meter("lr", train_tools.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Distillation Epoch: [{epoch}]"
    
    accum_iter = args.accum_iter
    optimizer.zero_grad()
    
    # Accumulatori per encoder metrics
    total_samples = 0
    sum_loss = 0.0
    sum_mse = 0.0
    sum_cos = 0.0
    sum_cos_sim = 0.0
    sum_mean_diff = 0.0
    sum_std_diff = 0.0
    
    # Accumulatori per decoder metrics
    sum_decoder_loss_total = 0.0
    sum_decoder_loss_masks = 0.0
    sum_decoder_loss_iou = 0.0
    sum_decoder_loss_tokens = 0.0
    
    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        
        epoch_f = epoch + data_iter_step / max(1, len(data_loader))
        
        # Get data
        image_paths = batch["image_paths"]
        
        if teacher_extractor is None:
            raise ValueError("teacher_extractor required but not provided")
        
        pil_images = batch["pil_images"]
        with torch.no_grad():
            # PASSA multi_view per coerenza intra-scena
            teacher_features = teacher_extractor(
                pil_images, 
                multi_view=args.multi_view_mode,
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

        # Compute encoder loss
        loss, loss_details = criterion(student_features, teacher_features, mse_type=args.mse_type)

        # ===== DECODER DISTILLATION =====
        if sam_prompt_encoder_teacher is not None:
            # Generate point prompts for decoder
            point_coords, point_labels = _generate_point_prompts_grid(
                batch_size=teacher_features.shape[0],
                image_size=1024,
                points_per_side=getattr(args, 'amg_points_per_side', 3),
                device=device,
            )
            
            # Get prompt embeddings from teacher (frozen)
            with torch.no_grad():
                sparse_emb, dense_emb = sam_prompt_encoder_teacher(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=None,
                )
                image_pe = sam_prompt_encoder_teacher.get_dense_pe().to(
                    device=teacher_features.device,
                    dtype=teacher_features.dtype
                )
                
                # Teacher decoder forward (frozen)
                t_masks, t_iou, t_tokens, t_obj = sam_mask_decoder_teacher(
                    image_embeddings=teacher_features,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=None,
                )
            
            # Student decoder forward (trainable)
            s_masks, s_iou, s_tokens, s_obj = sam_mask_decoder_student(
                image_embeddings=student_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                repeat_image=False,
                high_res_features=None,
            )
            
            # Compute decoder losses
            decoder_loss, decoder_loss_details = decoder_criterion(
                student_masks=s_masks,
                student_iou=s_iou,
                student_tokens=s_tokens,
                teacher_masks=t_masks,
                teacher_iou=t_iou,
                teacher_tokens=t_tokens,
            )
            
            # Combine losses
            decoder_weight = getattr(args, 'decoder_loss_weight', 1.0)

            # ========== PRINT DETTAGLIATO PRIMA ITERAZIONE ==========
            # if data_iter_step == 0 and epoch == 0:
            #     print("\n" + "="*80)
            #     print("FIRST BATCH LOSS BREAKDOWN")
            #     print("="*80)
            #     print(f"[ENCODER] MSE: {loss_details['mse_loss']:.6f}, Cosine: {loss_details['cos_loss']:.6f}, Total: {loss.item():.6f}")
            #     print(f"[DECODER] Masks: {decoder_loss_details['decoder_loss_masks']:.6f}, "
            #         f"IoU: {decoder_loss_details['decoder_loss_iou']:.6f}, "
            #         f"Tokens: {decoder_loss_details['decoder_loss_tokens']:.6f}")
            #     print(f"[DECODER] Total (before weight): {decoder_loss.item():.6f}")
            #     print(f"[DECODER] Weight multiplier: {decoder_weight}")
            #     print(f"[DECODER] Weighted total: {(decoder_weight * decoder_loss).item():.6f}")
            #     print(f"[COMBINED] Final loss: {(loss + decoder_weight * decoder_loss).item():.6f}")
            #     print("="*80 + "\n")

            loss = loss + decoder_weight * decoder_loss
            loss_details.update(decoder_loss_details)

        # Extract metrics
        mse_value = float(loss_details.get("enc_mse_loss", 0.0))
        cos_value = float(loss_details.get("enc_cos_loss", 0.0))
        cos_sim_value = float(loss_details.get("enc_cos_sim", 0.0))
        
        # Decoder metrics (0.0 se decoder non attivo)
        decoder_loss_total = float(loss_details.get("decoder_loss_total", 0.0))
        decoder_loss_masks = float(loss_details.get("decoder_loss_masks", 0.0))
        decoder_loss_iou = float(loss_details.get("decoder_loss_iou", 0.0))
        decoder_loss_tokens = float(loss_details.get("decoder_loss_tokens", 0.0))

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
                log_dict = {
                    # Encoder metrics (batch-level)
                    "train_encoder/enc_loss": float(loss_value),
                    "train_encoder/enc_mse_loss": float(mse_value),
                    "train_encoder/enc_cos_loss": float(cos_value),
                    "train_encoder/enc_cos_sim": float(cos_sim_value),
                    # Decoder metrics (batch-level)
                    "train_decoder/decoder_loss_total": float(decoder_loss_total),
                    "train_decoder/decoder_loss_masks": float(decoder_loss_masks),
                    "train_decoder/decoder_loss_iou": float(decoder_loss_iou),
                    "train_decoder/decoder_loss_tokens": float(decoder_loss_tokens),
                    # Progress
                    "epoch_progress": epoch_f,
                }
                # Aggiungi metriche decoder se presenti
                if decoder_loss_total > 0:
                    log_dict.update({
                        "train_decoder/lr": float(optimizer.param_groups[0]["lr"]),
                    })
                wandb.log(log_dict)
        
        # Gradient Accumulation
        loss /= accum_iter
        loss.backward()

        if (data_iter_step + 1) % accum_iter == 0:
            # print("\n" + "="*80)
            # print("GRAD CHECK - Sample gradient norms per group:")
            # for i, group in enumerate(optimizer.param_groups):
            #     grad_squares = [p.grad**2 for p in group['params'] if p.grad is not None]
            #     if len(grad_squares) > 0:
            #         grad_norm = torch.sqrt(sum((g.sum() for g in grad_squares)))
            #         group_name = ["encoder", "decoder", "transformer", "dino"][i] if i < 4 else f"group_{i}"
            #         print(f"  [{group_name}] Grad norm: {grad_norm.item():.6e}")
            #     else:
            #         group_name = ["encoder", "decoder", "transformer", "dino"][i] if i < 4 else f"group_{i}"
            #         print(f"  [{group_name}] Grad norm: 0.0 (no gradients)")
            # print("="*80 + "\n")
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad()

            # ========== PRINT LR DOPO PRIMO STEP ==========
            # if data_iter_step == 0 and epoch == 0:
            #     print("\n" + "="*80)
            #     print("LEARNING RATES AFTER FIRST OPTIMIZER STEP")
            #     print("="*80)
            #     for i, group in enumerate(optimizer.param_groups):
            #         group_name = ["encoder", "decoder", "transformer", "dino"][i] if i < 4 else f"group_{i}"
            #         print(f"  [{group_name}] Current LR: {group['lr']:.6e}")
            #     print("="*80 + "\n")

        # Accumulate weighted sums
        batch_size = student_features.shape[0]
        total_samples += batch_size
        sum_loss += loss_value * batch_size
        sum_mse += mse_value * batch_size
        sum_cos += cos_value * batch_size
        sum_cos_sim += cos_sim_value * batch_size
        sum_mean_diff += md * batch_size
        sum_std_diff += sd * batch_size
        
        # Accumulate decoder metrics
        sum_decoder_loss_total += decoder_loss_total * batch_size
        sum_decoder_loss_masks += decoder_loss_masks * batch_size
        sum_decoder_loss_iou += decoder_loss_iou * batch_size
        sum_decoder_loss_tokens += decoder_loss_tokens * batch_size

        # Clean up
        del loss, student_features, teacher_features
        
        # Update metrics
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss_value, **loss_details)

    # Return averaged stats
    denom = max(1, total_samples)
    results = {
        "train_enc_loss": sum_loss / denom,
        "train_enc_mse_loss": sum_mse / denom,
        "train_enc_cos_loss": sum_cos / denom,
        "train_enc_cos_sim": sum_cos_sim / denom,
        "train_enc_mean_diff": sum_mean_diff / denom,
        "train_enc_std_diff": sum_std_diff / denom,
        "train_decoder_loss_total": sum_decoder_loss_total / denom,
        "train_decoder_loss_masks": sum_decoder_loss_masks / denom,
        "train_decoder_loss_iou": sum_decoder_loss_iou / denom,
        "train_decoder_loss_tokens": sum_decoder_loss_tokens / denom,
        "train_loss_total": (sum_loss + sum_decoder_loss_total) / denom,
        "lr": optimizer.param_groups[0]["lr"],
        "samples": total_samples,
    }
    return results

@torch.no_grad()
def validate_one_epoch_distillation(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    decoder_criterion: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    args,
    teacher_extractor: Optional[TeacherFeatureExtractor] = None,
    sam_prompt_encoder_teacher=None,
    sam_mask_decoder_teacher=None,
    sam_mask_decoder_student=None,
) -> Dict:
    """
    Validate the model for one epoch on distillation task.
    
    Args:
        model: MapAnything model with dpt_feature_head_2
        criterion: DistillationLoss instance for encoder
        decoder_criterion: DecoderDistillationLoss instance for decoder
        data_loader: DataLoader providing validation data
        device: Device to run on
        epoch: Current epoch number
        args: Configuration namespace
        teacher_extractor: TeacherFeatureExtractor instance
        sam_prompt_encoder_teacher: Teacher prompt encoder (frozen)
        sam_mask_decoder_teacher: Teacher mask decoder (frozen)
        sam_mask_decoder_student: Student mask decoder (trainable)
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    metric_logger = train_tools.MetricLogger(delimiter=" | ")
    metric_logger.meters = defaultdict(lambda: train_tools.SmoothedValue(window_size=int(1e6)))
    header = f"Distillation Validation: [{epoch}]"
    
    # Accumulatori encoder
    total_samples = 0
    sum_loss = 0.0
    sum_mse = 0.0
    sum_cos = 0.0
    sum_cos_sim = 0.0
    sum_mean_diff = 0.0
    sum_std_diff = 0.0
    
    # Accumulatori decoder
    sum_decoder_loss_total = 0.0
    sum_decoder_loss_masks = 0.0
    sum_decoder_loss_iou = 0.0
    sum_decoder_loss_tokens = 0.0

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        
        image_paths = batch["image_paths"]
        
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
        
        # Compute encoder loss
        loss, loss_details = criterion(student_features, teacher_features, mse_type=args.mse_type)
        
        # ===== DECODER DISTILLATION =====
        if sam_prompt_encoder_teacher is not None:
            # Generate point prompts for decoder
            point_coords, point_labels = _generate_point_prompts_grid(
                batch_size=student_features.shape[0],
                image_size=1024,
                points_per_side=getattr(args, 'amg_points_per_side', 3),
                device=device,
            )
            
            # Get prompt embeddings from teacher (frozen)
            sparse_emb, dense_emb = sam_prompt_encoder_teacher(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            image_pe = sam_prompt_encoder_teacher.get_dense_pe().to(
                device=teacher_features.device,
                dtype=teacher_features.dtype
            )
            
            # Teacher decoder forward (frozen)
            t_masks, t_iou, t_tokens, t_obj = sam_mask_decoder_teacher(
                image_embeddings=teacher_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                repeat_image=False,
                high_res_features=None,
            )
            
            # Student decoder forward (no grad in validation)
            s_masks, s_iou, s_tokens, s_obj = sam_mask_decoder_student(
                image_embeddings=student_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                repeat_image=False,
                high_res_features=None,
            )
            
            # Compute decoder losses usando decoder_criterion
            decoder_loss, decoder_loss_details = decoder_criterion(
                student_masks=s_masks,
                student_iou=s_iou,
                student_tokens=s_tokens,
                teacher_masks=t_masks,
                teacher_iou=t_iou,
                teacher_tokens=t_tokens,
            )

            loss_details.update(decoder_loss_details)

            # Combine losses
            decoder_weight = getattr(args, 'decoder_loss_weight', 1.0)
            loss = loss + decoder_weight * decoder_loss
        
        # Extract metrics
        loss_value = loss.detach().cpu().item()
        mse_value = float(loss_details.get("enc_mse_loss", 0.0))
        cos_value = float(loss_details.get("enc_cos_loss", 0.0))
        cos_sim_value = float(loss_details.get("enc_cos_sim", 0.0))
        
        # Decoder metrics
        decoder_loss_total = float(loss_details.get("decoder_loss_total", 0.0))
        decoder_loss_masks = float(loss_details.get("decoder_loss_masks", 0.0))
        decoder_loss_iou = float(loss_details.get("decoder_loss_iou", 0.0))
        decoder_loss_tokens = float(loss_details.get("decoder_loss_tokens", 0.0))

        try:
            md, sd, cs = mean_std_difference(student_features, teacher_features)
            md = float(md)
            sd = float(sd)
            cs = float(cs)
        except Exception:
            md = sd = 0.0
            cs = cos_sim_value

        metric_logger.update(loss=loss_value, **loss_details)
        
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
        sum_mean_diff += md * batch_size
        sum_std_diff += sd * batch_size
        
        # Accumulate decoder metrics
        sum_decoder_loss_total += decoder_loss_total * batch_size
        sum_decoder_loss_masks += decoder_loss_masks * batch_size
        sum_decoder_loss_iou += decoder_loss_iou * batch_size
        sum_decoder_loss_tokens += decoder_loss_tokens * batch_size

        del student_features, teacher_features

    denom = max(1, total_samples)
    results = {
        "val_enc_loss": sum_loss / denom,
        "val_enc_mse_loss": sum_mse / denom,
        "val_enc_cos_loss": sum_cos / denom,
        "val_enc_cos_sim": sum_cos_sim / denom,
        "val_enc_mean_diff": sum_mean_diff / denom,
        "val_enc_std_diff": sum_std_diff / denom,
        "val_decoder_loss_total": sum_decoder_loss_total / denom,
        "val_decoder_loss_masks": sum_decoder_loss_masks / denom,
        "val_decoder_loss_iou": sum_decoder_loss_iou / denom,
        "val_decoder_loss_tokens": sum_decoder_loss_tokens / denom,
        "val_loss_total": (sum_loss + sum_decoder_loss_total) / denom,
        "samples": total_samples,
        "val_enc_loss_avg": sum_loss / denom,
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
    
    if args.print_args:
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

    # ========== LOAD MODEL ==========
    print("[INFO] Loading MapAnything model...")

    # Patch config file based on use_encoder_features
    if CONFIG_JSON_PATH and os.path.exists(CONFIG_JSON_PATH):
        with open(CONFIG_JSON_PATH, 'r') as f:
            config = json.load(f)
        
        # Modify input_feature_dims in feature_head_2 based on use_encoder_features
        if args.use_encoder_features:
            new_dims = [1024, 1024, 1024, 1024]
            print(f"[INFO] use_encoder_features=True: Setting feature_head_2.input_feature_dims to {new_dims}")
        else:
            new_dims = [1024, 768, 768, 768]
            print(f"[INFO] use_encoder_features=False: Setting feature_head_2.input_feature_dims to {new_dims}")
        
        if "pred_head_config" in config and "feature_head_2" in config["pred_head_config"]:
            config["pred_head_config"]["feature_head_2"]["input_feature_dims"] = new_dims
            
            # Save modified config back to original location to overwrite
            with open(CONFIG_JSON_PATH, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"[INFO] Overwritten config.json")

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
    print(f"[INFO] Model loaded. Has dpt_feature_head_2: {hasattr(model, 'dpt_feature_head_2')}")

    # ========== INITIALIZE STUDENT DECODER ==========
    print(f"[INFO] Building student MaskDecoder...")
    sam_mask_decoder_student = build_sam_mask_decoder(
        embed_dim=256,
        num_multimask_outputs=3,
        use_high_res_features=False,
        pred_obj_scores=False,
        pred_obj_scores_mlp=False,
        iou_prediction_use_sigmoid=False,
    ).to(device)
    print(f"[INFO] Student MaskDecoder built: {sum(p.numel() for p in sam_mask_decoder_student.parameters()):,} params")

    # Attach student decoder to model
    model_without_ddp.sam2_mask_decoder_student = sam_mask_decoder_student

    # ========== INITIALIZE TEACHER ENCODER ==========
    print(f"[INFO] Preparing teacher feature extractor...")
    teacher_extractor = None

    augment_cfg = {
        "enabled": getattr(args, "use_data_augmentation", True),
        "p_color_jitter": 0.75,     # 75% probabilità
        "p_blur": 0.05,              # 5% probabilità
        "p_grayscale": 0.05,         # 5% probabilità
    }
    teacher_extractor = TeacherFeatureExtractor(
        checkpoint_path=SAM2_PATH,
        device=str(device),
        augment_cfg=augment_cfg,
    )
    teacher_extractor.to(device)

    # ========== INITIALIZE TEACHER DECODER ==========
    print(f"[INFO] Loading teacher PromptEncoder and MaskDecoder...")
    sam_prompt_encoder_teacher, sam_mask_decoder_teacher = load_sam2_teacher_prompt_and_decoder(
        checkpoint_path=SAM2_PATH,
        device=str(device),
        image_size=1024,
        backbone_stride=16,
        embed_dim=256,
    )
    print(f"[INFO] Teacher decoder components loaded and frozen")

    # ========== BUILD DATALOADERS ==========
    
    # --- 1. TRAIN DATALOADER ---
    print(f"[INFO] Building train dataloader from {TRAIN_IMAGES_DIR}")
    train_image_paths = None
    
    # Logica Debug per SINGLE-VIEW: filtriamo la lista delle immagini PRIMA di creare il loader
    if args.debug_max_train_images and not args.multi_view_mode:
        all_imgs = sorted([
            os.path.join(TRAIN_IMAGES_DIR, f)
            for f in os.listdir(TRAIN_IMAGES_DIR)
            if DistillationDataset._is_image_file(f)
        ])
        train_image_paths = random.sample(all_imgs, min(args.debug_max_train_images, len(all_imgs)))
        print(f"[INFO] Single-View: Limited train to {len(train_image_paths)} IMAGES")
    
    data_loader_train = build_distillation_dataloader(
        image_dir=TRAIN_IMAGES_DIR,
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
    print(f"[INFO] Building val dataloader from {VAL_IMAGES_DIR}")
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

    # ========== FREEZE STRATEGY ==========
    # 1. Freeze tutto inizialmente
    print("[INFO] Freezing all parameters...")
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. Unfreeze dpt_feature_head_2 e sam2_compat (sempre trainable) e student MaskDecoder STUDENT ENCODER + DECODER
    print("[INFO] Unfreezing dpt_feature_head_2, sam2_compat, and sam2_mask_decoder_student...")
    for name, param in model.named_parameters():
        if name.startswith("dpt_feature_head_2") or name.startswith("sam2_compat") or name.startswith("sam2_mask_decoder_student"):
            param.requires_grad = True

    # 3. Unfreeze ultimi N blocchi di info_sharing.self_attention_blocks (opzionale)
    num_info_sharing_blocks = getattr(args, 'num_info_sharing_blocks_unfreeze', 0)
    if num_info_sharing_blocks > 0 and hasattr(model, "info_sharing"):
        info_sharing = model.info_sharing
        
        # Trova i blocchi (self_attention_blocks per MapAnything)
        if hasattr(info_sharing, "self_attention_blocks"):
            blocks = info_sharing.self_attention_blocks
        
        # Unfreeze gli ultimi N blocchi
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
            
            _ = verify_frozen_blocks(
                blocks, 
                block_name="Multi-View Transformer blocks",
                unfrozen_indices=unfrozen_indices
            )
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
            print("[DEBUG] Found 'encoder' in model")
            dino_encoder = model.encoder
        
        if dino_encoder is not None:
            # Cerca i blocchi transformer dell'encoder
            blocks = None
            if hasattr(dino_encoder, "model"):
                if hasattr(dino_encoder.model, "blocks"):
                    blocks = dino_encoder.model.blocks
            
            # Unfreeze gli ultimi N blocchi
            if blocks is not None and len(blocks) > 0:
                start_idx = max(0, len(blocks) - num_dino_layers_unfreeze)
                unfrozen_count = 0
                unfrozen_dino_indices = []
                for i in range(start_idx, len(blocks)):
                    for param in blocks[i].parameters():
                        param.requires_grad = True
                        unfrozen_count += param.numel()
                    unfrozen_dino_indices.append(i)
                # There are some additional layers wrt the 24 blocks that we have to unfreeze
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

                _ = verify_frozen_blocks(
                    blocks, 
                    block_name="DINOv2 encoder blocks",
                    unfrozen_indices=unfrozen_dino_indices
                )
            else:
                print("[WARN] DINOv2 encoder has no 'blocks' or 'layers' attribute. Skipping unfreezing.")
                args.dino_unfrozen_indices = []
        else:
            print("[WARN] DINOv2 encoder not found on model. Skipping unfreezing.")
            args.dino_unfrozen_indices = []
    else:
        args.dino_unfrozen_indices = []

    # ========== VERIFY TRAINABLE PARAMETERS ==========
    if args.print_trainable:
        print_trainable_summary(model, detailed=True)
    
    # Initialize criterion for STUDENT ENCODER distillation
    criterion = DistillationLoss(
        mse_weight=args.mse_weight,
        cosine_weight=args.cosine_weight,
        normalize=args.normalize_features,
    ).to(device)

    # Initialize criterion for STUDENT DECODER distillation
    decoder_criterion = DecoderDistillationLoss(
        weight_masks=args.decoder_masks_weight,
        weight_iou=args.decoder_iou_weight,
        weight_tokens=args.decoder_tokens_weight,
    ).to(device)

    # ========== VERIFICHE LOSS WEIGHTS ==========
    # print("\n" + "="*80)
    # print("LOSS WEIGHTS VERIFICATION")
    # print("="*80)
    # print(f"[ENCODER LOSS] MSE weight: {criterion.mse_weight}, Cosine weight: {criterion.cosine_weight}")
    # print(f"[DECODER LOSS] Masks: {decoder_criterion.weight_masks}, IoU: {decoder_criterion.weight_iou}, Tokens: {decoder_criterion.weight_tokens}")
    # print(f"[DECODER LOSS] Total weight multiplier: {getattr(args, 'decoder_loss_weight', 1.0)}")
    # print("="*80 + "\n")

    # ========== OPTIMIZER con LR differenziati ==========
    encoder_params = [] # STUDENT ENCODER (dpt_head_2 + sam2_compat)
    decoder_params = []  # STUDENT DECODER (MaskDecoder)
    transformer_params = [] # MULTI VIEW TRANSFORMER (info_sharing)
    dino_params = [] # DINOv2 ENCODER (encoder)
    other_params = [] # Fallback

    for name, p in model.named_parameters():
        if not p.requires_grad: # if frozen, skip
            continue
        if name.startswith("dpt_feature_head_2") or name.startswith("sam2_compat"):
            encoder_params.append(p)
        elif name.startswith("sam2_mask_decoder_student"):
            decoder_params.append(p)
        elif name.startswith("info_sharing"):
            transformer_params.append(p)
        elif name.startswith("encoder") and hasattr(args, 'dino_unfrozen_indices') and args.dino_unfrozen_indices:
            dino_params.append(p)
        else:
            other_params.append(p)

    # Fallback: se alcuni parametri trainabili non rientrano nelle categorie stampa un warning
    if other_params:
        print(f"[WARN] Found {len(other_params)} trainable parameters not matched to any group.")

    lr_encoder = args.lr * args.lr_encoder_scale
    lr_decoder = args.lr * args.lr_decoder_scale
    lr_dino = args.lr * args.lr_dino_scale
    lr_transformer = args.lr * args.lr_transformer_scale

    optimizer = optim.AdamW(
        [
            {"params": encoder_params, "lr": lr_encoder},
            {"params": decoder_params, "lr": lr_decoder},
            {"params": transformer_params, "lr": lr_transformer},
            {"params": dino_params, "lr": lr_dino},
        ],
        lr=args.lr,  # non usato per i gruppi espliciti, rimane come default
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    print(f"[INFO] Groups: encoder={sum(p.numel() for p in encoder_params):,} params @ LR {lr_encoder}, "
          f"decoder={sum(p.numel() for p in decoder_params):,} params @ LR {lr_decoder}, "
          f"dino={sum(p.numel() for p in dino_params):,} params @ LR {lr_dino}, "
          f"transformer={sum(p.numel() for p in transformer_params):,} params @ LR {lr_transformer}")
    
    # ========== VERIFICHE LEARNING RATES ==========
    # print("\n" + "="*80)
    # print("OPTIMIZER PARAM GROUPS VERIFICATION")
    # print("="*80)
    # print(f"Base LR: {args.lr}")
    # print(f"LR Scales: encoder={args.lr_encoder_scale}, decoder={args.lr_decoder_scale}, "
    #     f"transformer={args.lr_transformer_scale}, dino={args.lr_dino_scale}")
    # print("\nActual param_groups in optimizer:")
    # for i, group in enumerate(optimizer.param_groups):
    #     num_params = sum(p.numel() for p in group['params'])
    #     group_name = ["encoder", "decoder", "transformer", "dino"][i] if i < 4 else f"group_{i}"
    #     print(f"  [{group_name}] LR: {group['lr']:.6e}, Params: {num_params:,}, Weight decay: {group['weight_decay']}")
    # print("="*80 + "\n")
    
    # ========== WRAPPING IN DDP ==========
    if args.distributed.distributed:
        print("[INFO] Wrapping model in DistributedDataParallel (DDP)...")
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
    
    # ========== LEARNING RATE SCHEDULER ==========
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
        print(f"[INFO] Learning rate scheduler disabled. Base LR will remain constant at {args.lr}")
    
    # ========= RESUME FROM CHECKPOINT (IF SPECIFIED) ==========
    start_epoch = 0
    best_val_loss = float("inf")
    
    # Handle backward compatibility: if resume_ckpt is provided, load both encoder and decoder
    if args.resume_ckpt:
        print(f"[RESUME] Using legacy --resume_ckpt (loads both encoder and decoder): {args.resume_ckpt}")
        args.resume_encoder_ckpt = args.resume_ckpt
        args.resume_decoder_ckpt = args.resume_ckpt
    
    # Load encoder checkpoint if specified
    if args.resume_encoder_ckpt:
        enc_start_epoch, enc_best_val_loss = load_encoder_checkpoint(
            model_without_ddp=model_without_ddp,
            checkpoint_path=args.resume_encoder_ckpt,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
        )
        start_epoch = max(start_epoch, enc_start_epoch)
        best_val_loss = min(best_val_loss, enc_best_val_loss)
        
        # Handle LR override after encoder checkpoint load
        if args.lr_scheduler == "none" or args.override_lr:
            if len(optimizer.param_groups) >= 4:
                optimizer.param_groups[0]["lr"] = args.lr * args.lr_encoder_scale
                optimizer.param_groups[1]["lr"] = args.lr * args.lr_decoder_scale
                optimizer.param_groups[2]["lr"] = args.lr * args.lr_transformer_scale
                optimizer.param_groups[3]["lr"] = args.lr * args.lr_dino_scale
                print(
                    "[INFO] Overriding optimizer LR after encoder load: "
                    f"encoder={optimizer.param_groups[0]['lr']:.6e}, "
                    f"decoder={optimizer.param_groups[1]['lr']:.6e}, "
                    f"transformer={optimizer.param_groups[2]['lr']:.6e}, "
                    f"dino={optimizer.param_groups[3]['lr']:.6e}"
                )
    
    # Load decoder checkpoint if specified
    if args.resume_decoder_ckpt:
        try:
            dec_start_epoch, dec_best_val_loss = load_decoder_checkpoint(
                model_without_ddp=model_without_ddp,
                checkpoint_path=args.resume_decoder_ckpt,
                device=device,
                optimizer=None,
                scheduler=None,
                args=args,
            )
            start_epoch = max(start_epoch, dec_start_epoch)
            best_val_loss = min(best_val_loss, dec_best_val_loss)
            print(f"[RESUME] Decoder checkpoint loaded. Start epoch: {start_epoch}, Best val loss: {best_val_loss:.6f}")
        except Exception as e:
            print(f"[ERROR] Failed to load decoder checkpoint: {e}")
            raise
    
    # Load trainer checkpoint if specified
    if args.resume_trainer_ckpt:
        try:
            tr_start_epoch, tr_best_val_loss = load_trainer_checkpoint(
                checkpoint_path=args.resume_trainer_ckpt,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
            )
            start_epoch = max(start_epoch, tr_start_epoch)
            best_val_loss = min(best_val_loss, tr_best_val_loss)
            print(f"[RESUME] Trainer checkpoint loaded. Start epoch: {start_epoch}, Best val loss: {best_val_loss:.6f}")
        except Exception as e:
            print(f"[ERROR] Failed to load trainer checkpoint: {e}")
            raise
    
    # Handle LR override after loading checkpoints
    if args.lr_scheduler == "none" or args.override_lr:
        if len(optimizer.param_groups) >= 4:
            optimizer.param_groups[0]["lr"] = args.lr * args.lr_encoder_scale
            optimizer.param_groups[1]["lr"] = args.lr * args.lr_decoder_scale
            optimizer.param_groups[2]["lr"] = args.lr * args.lr_transformer_scale
            optimizer.param_groups[3]["lr"] = args.lr * args.lr_dino_scale
            print(
                "[INFO] Overriding optimizer LR: "
                f"encoder={optimizer.param_groups[0]['lr']:.6e}, "
                f"decoder={optimizer.param_groups[1]['lr']:.6e}, "
                f"transformer={optimizer.param_groups[2]['lr']:.6e}, "
                f"dino={optimizer.param_groups[3]['lr']:.6e}"
            )
    
    # Scheduler advance logic if we resumed from a checkpoint
    if args.resume_encoder_ckpt or args.resume_decoder_ckpt or args.resume_trainer_ckpt:
        if args.override_scheduler and scheduler is not None:
            resumed_epoch = start_epoch
            
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
        
        if start_epoch > 0:
            print(f"[RESUME] Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
    
    # ========== INITIALIZE WANDB (only if rank == 0) ==========
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
    print(f"[INFO] Start distillation training for {args.epochs} epochs from epoch {start_epoch}")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        if args.distributed.distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
        
        epoch_start = time.time()
        
        # Train one epoch
        train_stats = train_one_epoch_distillation(
            model=model,
            criterion=criterion,
            decoder_criterion=decoder_criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            teacher_extractor=teacher_extractor,
            sam_prompt_encoder_teacher=sam_prompt_encoder_teacher,
            sam_mask_decoder_teacher=sam_mask_decoder_teacher,
            sam_mask_decoder_student=sam_mask_decoder_student,
        )
        
        # Validation
        val_stats = {}
        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0:
            val_stats = validate_one_epoch_distillation(
            model=model,
            criterion=criterion,
            decoder_criterion=decoder_criterion,
            data_loader=data_loader_val,
            device=device,
            epoch=epoch,
            args=args,
            teacher_extractor=teacher_extractor,
            sam_prompt_encoder_teacher=sam_prompt_encoder_teacher,
            sam_mask_decoder_teacher=sam_mask_decoder_teacher,
            sam_mask_decoder_student=sam_mask_decoder_student,
        )
            
            # Check for new best
            val_loss_avg = val_stats.get("loss_avg", float("inf"))
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                print(f"New best validation loss: {best_val_loss:.6f}")
                # Save best checkpoint
                if global_rank == 0:
                    # Default: save encoder, decoder, and trainer separately
                    if not args.save_combined_ckpt:
                        if args.save_encoder_ckpt:
                            save_encoder_checkpoint(
                                model_without_ddp,
                                optimizer,
                                scheduler,
                                epoch,
                                best_val_loss,
                                args.output_dir,
                                tag="best",
                                args=args,
                            )
                        if args.save_decoder_ckpt:
                            save_decoder_checkpoint(
                                model_without_ddp,
                                optimizer,
                                scheduler,
                                epoch,
                                best_val_loss,
                                args.output_dir,
                                tag="best",
                                args=args,
                            )
                        if getattr(args, "save_trainer_ckpt", True):
                            save_trainer_checkpoint(
                                optimizer,
                                scheduler,
                                epoch,
                                best_val_loss,
                                args.output_dir,
                                tag="best",
                            )
                    # Legacy: save both in single file if --save_combined_ckpt is set
                    else:
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
                # Default: save encoder, decoder, and trainer separately
                if not args.save_combined_ckpt:
                    if args.save_encoder_ckpt:
                        save_encoder_checkpoint(
                            model_without_ddp,
                            optimizer,
                            scheduler,
                            epoch,
                            best_val_loss,
                            args.output_dir,
                            tag=f"epoch{epoch+1}",
                            args=args,
                        )
                    if args.save_decoder_ckpt:
                        save_decoder_checkpoint(
                            model_without_ddp,
                            optimizer,
                            scheduler,
                            epoch,
                            best_val_loss,
                            args.output_dir,
                            tag=f"epoch{epoch+1}",
                            args=args,
                        )
                    if getattr(args, "save_trainer_ckpt", True):
                        save_trainer_checkpoint(
                            optimizer,
                            scheduler,
                            epoch,
                            best_val_loss,
                            args.output_dir,
                            tag=f"epoch{epoch+1}",
                        )
                # Legacy: save both in single file if --save_combined_ckpt is set
                else:
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

        # Log to wandb
        if args.use_wandb and WANDB_AVAILABLE and global_rank == 0:
            log_dict = {
                "epoch": epoch + 1,
                # Encoder metrics (train)
                "test_encoder/train_enc_loss": train_stats.get("train_enc_loss", 0.0),
                "test_encoder/train_enc_mse_loss": train_stats.get("train_enc_mse_loss", 0.0),
                "test_encoder/train_enc_cos_loss": train_stats.get("train_enc_cos_loss", 0.0),
                "test_encoder/train_enc_mean_diff": train_stats.get("train_enc_mean_diff", 0.0),
                "test_encoder/train_enc_std_diff": train_stats.get("train_enc_std_diff", 0.0),
                "test_encoder/train_enc_cos_sim": train_stats.get("train_enc_cos_sim", 0.0),
                # Decoder metrics (train)
                "test_decoder/train_decoder_loss_total": train_stats.get("train_decoder_loss_total", 0.0),
                "test_decoder/train_decoder_loss_masks": train_stats.get("train_decoder_loss_masks", 0.0),
                "test_decoder/train_decoder_loss_iou": train_stats.get("train_decoder_loss_iou", 0.0),
                "test_decoder/train_decoder_loss_tokens": train_stats.get("train_decoder_loss_tokens", 0.0),
                # Total loss and metrics (train)
                "test_totale/train_loss_total": train_stats.get("train_loss_total", 0.0),
                "test_totale/lr": optimizer.param_groups[0]["lr"],
                "test_totale/epoch_time_sec": epoch_time,
            }
            if val_stats:
                log_dict.update({
                    # Encoder metrics (val)
                    "test_encoder/val_enc_loss": val_stats.get("val_enc_loss", 0.0),
                    "test_encoder/val_enc_mse_loss": val_stats.get("val_enc_mse_loss", 0.0),
                    "test_encoder/val_enc_cos_loss": val_stats.get("val_enc_cos_loss", 0.0),
                    "test_encoder/val_enc_mean_diff": val_stats.get("val_enc_mean_diff", 0.0),
                    "test_encoder/val_enc_std_diff": val_stats.get("val_enc_std_diff", 0.0),
                    "test_encoder/val_enc_cos_sim": val_stats.get("val_enc_cos_sim", 0.0),
                    # Decoder metrics (val)
                    "test_decoder/val_decoder_loss_total": val_stats.get("val_decoder_loss_total", 0.0),
                    "test_decoder/val_decoder_loss_masks": val_stats.get("val_decoder_loss_masks", 0.0),
                    "test_decoder/val_decoder_loss_iou": val_stats.get("val_decoder_loss_iou", 0.0),
                    "test_decoder/val_decoder_loss_tokens": val_stats.get("val_decoder_loss_tokens", 0.0),
                    # Total loss (val)
                    "test_totale/val_loss_total": val_stats.get("val_loss_total", 0.0),
                })
            wandb.log(log_dict)
        
        # Console print con decoder loss
        decoder_train_loss = train_stats.get("train_decoder_loss_total", 0.0)
        decoder_val_loss = val_stats.get("val_decoder_loss_total", 0.0) if val_stats else 0.0
        
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_stats.get('train_enc_loss', 0):.6f} "
            f"(Enc: {train_stats.get('train_enc_mse_loss', 0):.4f}, Dec: {decoder_train_loss:.4f}) | "
            f"Val Loss: {val_stats.get('val_enc_loss', 0):.6f} "
            f"(Enc: {val_stats.get('val_enc_mse_loss', 0) if val_stats else 0:.4f}, Dec: {decoder_val_loss:.4f}) | "
            f"Time: {epoch_time:.2f}s"
        )
    
    # Save final checkpoint
    if global_rank == 0:
        # Default: save encoder, decoder, and trainer separately
        if not args.save_combined_ckpt:
            if args.save_encoder_ckpt:
                save_encoder_checkpoint(
                    model_without_ddp,
                    optimizer,
                    scheduler,
                    args.epochs - 1,
                    best_val_loss,
                    args.output_dir,
                    tag="final",
                    args=args,
                )
            if args.save_decoder_ckpt:
                save_decoder_checkpoint(
                    model_without_ddp,
                    optimizer,
                    scheduler,
                    args.epochs - 1,
                    best_val_loss,
                    args.output_dir,
                    tag="final",
                    args=args,
                )
            if getattr(args, "save_trainer_ckpt", True):
                save_trainer_checkpoint(
                    optimizer,
                    scheduler,
                    args.epochs - 1,
                    best_val_loss,
                    args.output_dir,
                    tag="final",
                )
        # Legacy: save both in single file if --save_combined_ckpt is set
        else:
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

def save_trainer_checkpoint(
    optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    output_dir: str,
    tag: str = "last",
):
    """
    Save trainer state (optimizer/scheduler/epoch/best_val_loss) in a separate file.
    
    Args:
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        best_val_loss: Best validation loss so far
        output_dir: Directory to save checkpoint
        tag: Tag for checkpoint filename (e.g., "best", "last", "epoch10")
    """
    state = {
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    
    if WANDB_AVAILABLE and wandb.run is not None:
        state["wandb_run_id"] = wandb.run.id
    
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_trainer_{tag}.pth"
    torch.save(state, ckpt_path)
    print(f"[SAVE] Trainer checkpoint saved: {ckpt_path}")


def load_trainer_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    optimizer=None,
    scheduler=None,
    args=None,
) -> Tuple[int, float]:
    """
    Load trainer state (optimizer/scheduler) from checkpoint.
    
    Args:
        checkpoint_path: Path to trainer checkpoint
        device: Device to load checkpoint on
        optimizer: Optimizer (required for loading optimizer state)
        scheduler: Scheduler (optional, for loading scheduler state)
        args: Training arguments with override flags
    
    Returns:
        (start_epoch, best_val_loss) tuple from checkpoint
    """
    print(f"[LOAD] Loading trainer checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if optimizer is not None and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("[INFO] Loaded optimizer state from trainer checkpoint")
        except Exception as e:
            print(f"[WARN] Failed loading optimizer state: {e}")
    
    if (
        scheduler is not None
        and "scheduler" in ckpt
        and not getattr(args, "override_scheduler", False)
    ):
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
            print("[INFO] Loaded scheduler state from trainer checkpoint")
        except Exception as e:
            print(f"[WARN] Failed loading scheduler state: {e}")
    
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    return start_epoch, best_val_loss


def load_encoder_checkpoint(
    model_without_ddp,
    checkpoint_path: str,
    device: torch.device,
    optimizer=None,
    scheduler=None,
    args=None,
) -> Tuple[int, float]:
    """
    Load checkpoint containing only ENCODER trainable components.
    
    Loads:
        - dpt_feature_head_2 (student encoder)
        - sam2_compat (student encoder compatibility layer)
        - unfrozen info_sharing blocks (multi-view transformer)
        - unfrozen DINOv2 encoder blocks
        - optimizer state (if provided)
        - scheduler state (if provided and not overridden)
    
    Args:
        model_without_ddp: Model without DDP wrapper
        checkpoint_path: Path to encoder checkpoint
        device: Device to load checkpoint on
        optimizer: Optimizer (optional, for loading optimizer state)
        scheduler: Scheduler (optional, for loading scheduler state)
        args: Training arguments with override flags
    
    Returns:
        (start_epoch, best_val_loss) tuple from checkpoint
    """
    print(f"[LOAD] Loading encoder checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load encoder head
    if "dpt_feature_head_2" in ckpt:
        model_without_ddp.dpt_feature_head_2.load_state_dict(ckpt["dpt_feature_head_2"])
        print("[INFO] Loaded dpt_feature_head_2 state from encoder checkpoint")
    else:
        print("[WARN] dpt_feature_head_2 not found in encoder checkpoint!")
    
    # Load sam2_compat if present
    if "sam2_compat" in ckpt and hasattr(model_without_ddp, "sam2_compat"):
        model_without_ddp.sam2_compat.load_state_dict(ckpt["sam2_compat"])
        print("[INFO] Loaded sam2_compat state from encoder checkpoint")
    elif hasattr(model_without_ddp, "sam2_compat"):
        print("[INFO] sam2_compat not in encoder checkpoint; using random initialization")
    
    # Load unfrozen info_sharing blocks
    if "info_sharing_blocks" in ckpt and hasattr(model_without_ddp, "info_sharing"):
        info = model_without_ddp.info_sharing
        blocks = getattr(info, "self_attention_blocks", None)
        if blocks is None:
            print("[WARN] info_sharing has no self_attention_blocks. Skipping restore.")
            blocks = []
        
        saved_indices = ckpt.get("info_sharing_unfrozen_indices", [])
        for idx, state_dict in ckpt["info_sharing_blocks"].items():
            if idx < len(blocks):
                try:
                    blocks[idx].load_state_dict(state_dict)
                except Exception as e:
                    print(f"[WARN] Failed loading info_sharing block {idx}: {e}")
        
        print(f"[INFO] Restored {len(saved_indices)} unfrozen info_sharing blocks from encoder checkpoint")
        if args is not None:
            args.info_sharing_unfrozen_indices = saved_indices
        
        # Load wrapper params if present
        if "info_sharing_wrappers" in ckpt:
            for name, data in ckpt["info_sharing_wrappers"].items():
                try:
                    param = dict(model_without_ddp.info_sharing.named_parameters())[name]
                    param.data.copy_(data)
                    param.requires_grad = True
                except Exception as e:
                    print(f"[WARN] Failed loading info_sharing wrapper param {name}: {e}")
            print(f"[INFO] Restored {len(ckpt['info_sharing_wrappers'])} info_sharing wrapper params")
    
    # Load unfrozen DINOv2 blocks
    if "dino_encoder_blocks" in ckpt and hasattr(model_without_ddp, "encoder"):
        dino_encoder = model_without_ddp.encoder
        dino_model = dino_encoder.model if hasattr(dino_encoder, "model") else dino_encoder
        blocks = getattr(dino_model, "blocks", None)
        if blocks is None:
            print("[WARN] DINO model has no .blocks. Skipping restore.")
            blocks = []
        
        for idx, state_dict in ckpt["dino_encoder_blocks"].items():
            if idx < len(blocks):
                try:
                    blocks[idx].load_state_dict(state_dict)
                except Exception as e:
                    print(f"[WARN] Failed loading DINOv2 block {idx}: {e}")
        
        print(f"[INFO] Restored {len(ckpt['dino_encoder_blocks'])} unfrozen DINOv2 encoder blocks")
        if args is not None:
            args.dino_unfrozen_indices = list(ckpt["dino_encoder_blocks"].keys())
        
        # Load wrapper params if present
        if "dino_encoder_wrappers" in ckpt:
            for name, data in ckpt["dino_encoder_wrappers"].items():
                try:
                    param = dict(dino_model.named_parameters())[name]
                    param.data.copy_(data)
                    param.requires_grad = True
                except Exception as e:
                    print(f"[WARN] Failed loading DINOv2 wrapper param {name}: {e}")
            print(f"[INFO] Restored {len(ckpt['dino_encoder_wrappers'])} DINOv2 wrapper params")
    
    # NOTE: optimizer/scheduler are NOT loaded from encoder checkpoint
    # Use load_trainer_checkpoint() instead
    
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    
    return start_epoch, best_val_loss


def load_decoder_checkpoint(
    model_without_ddp,
    checkpoint_path: str,
    device: torch.device,
    optimizer=None,
    scheduler=None,
    args=None,
) -> Tuple[int, float]:
    """
    Load checkpoint containing only DECODER trainable components.
    
    Loads:
        - sam2_mask_decoder_student (student decoder)
        - optimizer state (if provided)
        - scheduler state (if provided and not overridden)
    
    Args:
        model_without_ddp: Model without DDP wrapper
        checkpoint_path: Path to decoder checkpoint
        device: Device to load checkpoint on
        optimizer: Optimizer (optional, for loading optimizer state)
        scheduler: Scheduler (optional, for loading scheduler state)
        args: Training arguments with override flags
    
    Returns:
        (start_epoch, best_val_loss) tuple from checkpoint
    """
    print(f"[LOAD] Loading decoder checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load decoder
    if "sam2_mask_decoder_student" in ckpt and hasattr(model_without_ddp, "sam2_mask_decoder_student"):
        model_without_ddp.sam2_mask_decoder_student.load_state_dict(ckpt["sam2_mask_decoder_student"])
        print("[INFO] Loaded sam2_mask_decoder_student state from decoder checkpoint")
    elif hasattr(model_without_ddp, "sam2_mask_decoder_student"):
        print("[WARN] sam2_mask_decoder_student not found in decoder checkpoint; using random initialization")
    else:
        print("[WARN] sam2_mask_decoder_student not present in model!")
    
    # NOTE: optimizer/scheduler are NOT loaded from decoder checkpoint
    # Use load_trainer_checkpoint() instead
    
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    
    return start_epoch, best_val_loss


def save_encoder_checkpoint(
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
    Save checkpoint containing only ENCODER trainable components.
    
    Saves:
        - dpt_feature_head_2 (student encoder)
        - sam2_compat (student encoder compatibility layer)
        - unfrozen info_sharing blocks (multi-view transformer)
        - unfrozen DINOv2 encoder blocks
        - training metadata (epoch, best_val_loss, wandb_run_id)
    
    NOTE: optimizer/scheduler are NOT saved here.
    Use save_trainer_checkpoint() for optimizer/scheduler state.
    
    Args:
        model_without_ddp: Model without DDP wrapper
        optimizer: Optimizer (unused, kept for backward compatibility)
        scheduler: Learning rate scheduler (unused, kept for backward compatibility)
        epoch: Current epoch
        best_val_loss: Best validation loss so far
        output_dir: Directory to save checkpoint
        tag: Tag for checkpoint filename (e.g., "best", "last", "epoch10")
        args: Training arguments with unfrozen indices info
    """
    state = {
        "dpt_feature_head_2": model_without_ddp.dpt_feature_head_2.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }

    # Save sam2_compat if present
    if hasattr(model_without_ddp, "sam2_compat"):
        state["sam2_compat"] = model_without_ddp.sam2_compat.state_dict()
        print("[INFO] Added sam2_compat to encoder checkpoint")

    # Save unfrozen info_sharing blocks if any
    if args is not None and hasattr(model_without_ddp, "info_sharing") and getattr(args, "info_sharing_unfrozen_indices", []):
        info = model_without_ddp.info_sharing
        blocks = getattr(info, "self_attention_blocks", None)
        if blocks is None:
            blocks = []
        indices = [i for i in getattr(args, "info_sharing_unfrozen_indices", []) if i < len(blocks)]
        state["info_sharing_unfrozen_indices"] = indices
        state["info_sharing_blocks"] = {i: blocks[i].state_dict() for i in indices}
        print(f"[INFO] Added {len(indices)} unfrozen info_sharing blocks to encoder checkpoint: {indices}")

        if args.num_info_sharing_blocks_unfreeze == 24:
            wrapper_state = {}
            for name, param in model_without_ddp.info_sharing.named_parameters():
                if name.startswith(("proj_embed", "norm")):
                    wrapper_state[name] = param.data.clone()
            if wrapper_state:
                state["info_sharing_wrappers"] = wrapper_state
                print(f"[INFO] Added {len(wrapper_state)} info_sharing wrapper params to encoder checkpoint")

    # Save unfrozen DINOv2 blocks
    if args is not None and hasattr(args, 'dino_unfrozen_indices') and args.dino_unfrozen_indices:
        dino_encoder = model_without_ddp.encoder
        dino_model = dino_encoder.model if hasattr(dino_encoder, "model") else dino_encoder
        blocks = getattr(dino_model, "blocks", None)
        if blocks is None:
            blocks = []
        
        state["dino_encoder_blocks"] = {}
        if blocks:
            for idx in args.dino_unfrozen_indices:
                if idx < len(blocks):
                    state["dino_encoder_blocks"][idx] = blocks[idx].state_dict()

        if args.num_dino_layers_unfreeze == 24:
            wrapper_state = {}
            dino_model = dino_encoder.model if hasattr(dino_encoder, "model") else dino_encoder
            for name, param in dino_model.named_parameters():
                if name.startswith(("patch_embed.proj", "pos_embed", "cls_token", "norm")):
                    wrapper_state[name] = param.data.clone()
            if wrapper_state:
                state["dino_encoder_wrappers"] = wrapper_state
                print(f"[INFO] Added {len(wrapper_state)} DINOv2 wrapper params to encoder checkpoint")

        print(f"[INFO] Added {len(state['dino_encoder_blocks'])} unfrozen DINOv2 encoder blocks to encoder checkpoint: {list(state['dino_encoder_blocks'].keys())}")
    
    # Save wandb run_id if available
    if WANDB_AVAILABLE and wandb.run is not None:
        state["wandb_run_id"] = wandb.run.id
    
    # Crea la sottocartella checkpoints
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = ckpt_dir / f"checkpoint_encoder_{tag}.pth"
    torch.save(state, ckpt_path)
    print(f"[SAVE] Encoder checkpoint saved: {ckpt_path}")


def save_decoder_checkpoint(
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
    Save checkpoint containing only DECODER trainable components.
    
    Saves:
        - sam2_mask_decoder_student (student decoder)
        - training metadata (epoch, best_val_loss, wandb_run_id)
    
    NOTE: optimizer/scheduler are NOT saved here.
    Use save_trainer_checkpoint() for optimizer/scheduler state.
    
    Args:
        model_without_ddp: Model without DDP wrapper
        optimizer: Optimizer (unused, kept for backward compatibility)
        scheduler: Learning rate scheduler (unused, kept for backward compatibility)
        epoch: Current epoch
        best_val_loss: Best validation loss so far
        output_dir: Directory to save checkpoint
        tag: Tag for checkpoint filename (e.g., "best", "last", "epoch10")
        args: Training arguments
    """
    state = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }

    # Save student mask decoder if present
    if hasattr(model_without_ddp, "sam2_mask_decoder_student"):
        state["sam2_mask_decoder_student"] = model_without_ddp.sam2_mask_decoder_student.state_dict()
        print("[INFO] Added sam2_mask_decoder_student to decoder checkpoint")
    else:
        print("[WARN] sam2_mask_decoder_student not found in model!")
    
    # Save wandb run_id if available
    if WANDB_AVAILABLE and wandb.run is not None:
        state["wandb_run_id"] = wandb.run.id
    
    # Crea la sottocartella checkpoints
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = ckpt_dir / f"checkpoint_decoder_{tag}.pth"
    torch.save(state, ckpt_path)
    print(f"[SAVE] Decoder checkpoint saved: {ckpt_path}")


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
    Save checkpoint containing trainable components for distillation.
    
    DEPRECATED: This function is kept for backward compatibility.
    Use save_encoder_checkpoint and save_decoder_checkpoint separately instead.
    
    Saves:
        - dpt_feature_head_2 (student encoder)
        - sam2_compat (student encoder compatibility layer)
        - sam2_mask_decoder_student (student decoder)
        - unfrozen info_sharing blocks (multi-view transformer)
        - unfrozen DINOv2 encoder blocks
        - optimizer state
        - scheduler state
        - training metadata (epoch, best_val_loss, wandb_run_id)
    
    Args:
        model_without_ddp: Model without DDP wrapper
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        best_val_loss: Best validation loss so far
        output_dir: Directory to save checkpoint
        tag: Tag for checkpoint filename (e.g., "best", "last", "epoch10")
        args: Training arguments with unfrozen indices info
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
        print("[INFO] Added sam2_compat to checkpoint")

    # Save unfrozen info_sharing blocks if any
    if args is not None and hasattr(model_without_ddp, "info_sharing") and getattr(args, "info_sharing_unfrozen_indices", []):
        info = model_without_ddp.info_sharing
        blocks = getattr(info, "self_attention_blocks", None)
        if blocks is None:
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
        dino_model = dino_encoder.model if hasattr(dino_encoder, "model") else dino_encoder
        blocks = getattr(dino_model, "blocks", None)
        if blocks is None:
            blocks = []
        
        state["dino_encoder_blocks"] = {}
        if blocks:
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

    # Save student mask decoder if present
    if hasattr(model_without_ddp, "sam2_mask_decoder_student"):
        state["sam2_mask_decoder_student"] = model_without_ddp.sam2_mask_decoder_student.state_dict()
        print("[INFO] Added sam2_mask_decoder_student to checkpoint")

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    
    # Save wandb run_id if available
    if WANDB_AVAILABLE and wandb.run is not None:
        state["wandb_run_id"] = wandb.run.id
    
    # Crea la sottocartella checkpoints
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = ckpt_dir / f"checkpoint_{tag}.pth"
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
    parser.add_argument("--use_data_augmentation", action="store_true", help="Enable data augmentation")
    
    # Multi-view
    parser.add_argument("--multi_view_mode", action="store_true", help="Enable multi-view mode (cross-attention between views)")
    parser.add_argument("--max_views", type=int, default=6, help="Max views per scene. Train: Random Sample. Val: First N.")
    
    # Checkpointing
    parser.add_argument("--resume_ckpt", type=str, default=None, help="[DEPRECATED] Path to checkpoint to resume from (loads both encoder and decoder)")
    parser.add_argument("--resume_encoder_ckpt", type=str, default=None, help="Path to encoder checkpoint to load")
    parser.add_argument("--resume_decoder_ckpt", type=str, default=None, help="Path to decoder checkpoint to load")
    parser.add_argument("--resume_trainer_ckpt", type=str, default=None, help="Path to trainer checkpoint (optimizer/scheduler) to load")
    parser.add_argument("--save_encoder_ckpt", action="store_false", default=True, help="Disable separate encoder checkpoint saving (saves by default)")
    parser.add_argument("--save_decoder_ckpt", action="store_false", default=True, help="Disable separate decoder checkpoint saving (saves by default)")
    parser.add_argument("--save_trainer_ckpt", action="store_false", default=True, help="Disable separate trainer checkpoint saving (saves by default)")
    parser.add_argument("--save_combined_ckpt", action="store_true", default=False, help="Save encoder and decoder in a single combined checkpoint file (legacy behavior)")
    parser.add_argument("--save_freq", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--eval_freq", type=int, default=1, help="Run validation every N epochs")
    
    # Logging
    parser.add_argument("--print_freq", type=int, default=10, help="Print frequency (iterations)")
    parser.add_argument("--print_trainable", action="store_true", help="Print detailed trainable parameters and exit")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="mapanything-distillation", help="W&B project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_resume_id", type=str, default=None, help="W&B run ID to resume")
    parser.add_argument("--save_visualizations", action="store_true", help="Save PCA visualizations during validation")
    parser.add_argument("--log_freq", type=int, default=100, help="Log to W&B every N batches")
    parser.add_argument("--print_args", action="store_true", help="Print all arguments before starting distillation")
    
    # Distributed (opzionale): abilita DDP; dist_url di solito 'env://' con torchrun; local_rank impostato da torchrun
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--dist_url", type=str, default="env://", help="URL for distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")

    # Unfreeze strategy
    parser.add_argument("--num_info_sharing_blocks_unfreeze", type=int, default=0, help="Number of last info_sharing transformer blocks to unfreeze") # max 24
    parser.add_argument("--num_dino_layers_unfreeze", type=int, default=0, help="Number of last DINOv2 encoder layers to unfreeze") # max 24

    # Learning rates for different parts
    parser.add_argument("--lr_encoder_scale", type=float, default=1.0, help="Scale factor for STUDENT ENCODER LR")
    parser.add_argument("--lr_decoder_scale", type=float, default=1.0, help="Scale factor for STUDENT DECODER LR")
    parser.add_argument("--lr_dino_scale", type=float, default=0.1, help="Scale factor for DINOv2 encoder LR")
    parser.add_argument("--lr_transformer_scale", type=float, default=1.0, help="Scale factor MULTI-VIEW TRANSFORMER LR")

    # Decoder distillation
    parser.add_argument("--decoder_loss_weight", type=float, default=1.0, help="Weight for total decoder loss")
    parser.add_argument("--decoder_tokens_weight", type=float, default=1.0, help="Weight for tokens loss component")
    parser.add_argument("--decoder_masks_weight", type=float, default=0.5, help="Weight for masks loss component")
    parser.add_argument("--decoder_iou_weight", type=float, default=0.3, help="Weight for IoU loss component")
    parser.add_argument("--amg_points_per_side", type=int, default=3, help="Points per side for decoder prompt grid")
    
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
