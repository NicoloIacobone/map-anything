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

# Enable TF32 precision if supported
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True

def setup_runtime_paths(args):
    """Inizializza OUT_DIR, BASE_DIR, DATASET, SAM2_PATH e le directory immagini/feature usando args."""
    import torch.hub as _torch_hub
    global OUT_DIR, BASE_DIR, DATASET, SAM2_PATH
    global TRAIN_SPLIT, VAL_SPLIT
    global IMAGES_DIRNAME, FEATURES_DIRNAME
    global TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_FEATURES_DIR, VAL_FEATURES_DIR
    global run_cluster

    run_cluster = not sys.stdout.isatty()
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
    else:
        OUT_DIR = "/scratch2/nico/distillation/output"
        BASE_DIR = "/scratch2/nico/distillation/dataset"
        SAM2_PATH = "/scratch2/nico/sam2/checkpoints/sam2.1_hiera_large.pt"

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

    print(f"[INFO] Using TRAIN_IMAGES_DIR: {TRAIN_IMAGES_DIR}")
    print(f"[INFO] Using VAL_IMAGES_DIR: {VAL_IMAGES_DIR}")
    print(f"[INFO] Using TRAIN_FEATURES_DIR: {TRAIN_FEATURES_DIR}")
    print(f"[INFO] Using VAL_FEATURES_DIR: {VAL_FEATURES_DIR}")

def get_augmentation(no_augmentation: bool):
    """
    Definisce la pipeline di data augmentation.
    Target resolution: 518x518 (MapAnything standard/DINOv2 friendly).
    """
    if no_augmentation:
        # Nessuna augmentation: solo resize deterministico e conversione
        return transforms.Compose([
            transforms.Resize((518, 518)),
            # Nota: Non convertiamo in Tensor qui, lasciamo PIL per coerenza col Teacher
        ])
    
    # Augmentation completa (VGGT/MapAnything style)
    return transforms.Compose([
        # Random Aspect Ratio & Scale (include resize)
        transforms.RandomResizedCrop(518, scale=(0.33, 1.0), ratio=(0.75, 1.333)),
        # Photometric
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
    ])

# ==================== Dataset Classes ====================
class DistillationDataset(Dataset):
    """
    Dataset per la distillazione: supporta Single-View (lista piatta) e Multi-View (scene folders).
    """
    
    def __init__(
        self,
        image_dir: str,
        features_dir: Optional[str] = None,
        teacher_extractor: Optional[callable] = None,
        image_paths: Optional[List[str]] = None,
        transform=None,
        multi_view_mode: bool = False,
        max_views_per_scene: int = 6,
        split: str = "train",
        no_augmentation: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.features_dir = Path(features_dir) if features_dir else None
        self.teacher_extractor = teacher_extractor
        self.transform = transform
        self.multi_view_mode = multi_view_mode
        self.max_views_per_scene = max_views_per_scene
        self.is_train = "train" in split.lower()
        self.transform = get_augmentation(no_augmentation)
        
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
                    img = Image.open(p).convert("RGB")

                    if self.transform is not None:
                        img = self.transform(img)

                    pil_images_list.append(img)
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
    """
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        from feature_extractor import load_sam2_feature_extractor
        self.extractor = load_sam2_feature_extractor(checkpoint_path, device)
        self.device = device
        print(f"[Teacher] Loaded SAM2 feature extractor on {device}")
    
    @torch.no_grad()
    def __call__(self, pil_images: List) -> torch.Tensor:
        """
        Estrae feature da lista di PIL images.
        
        Args:
            pil_images: Lista di PIL.Image objects
        
        Returns:
            torch.Tensor (B, 256, 64, 64)
        """
        features = []
        for pil_img in pil_images:
            feat = self.extractor(pil_img)  # (1, 256, 64, 64) o (256, 64, 64)
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat)
            if feat.ndim == 3:
                feat = feat.unsqueeze(0)  # (1, 256, 64, 64)
            features.append(feat)
        
        return torch.cat(features, dim=0)  # (B, 256, 64, 64)
    
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
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Calcola la loss di distillazione.
        
        Args:
            student_features: Tensor (B,C,H,W) dallo studente
            teacher_features: Tensor (B,C,H,W) dal teacher
        
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

        mse_per_sample = F.mse_loss(
            student_norm, 
            teacher_norm, 
            reduction='none'  # (B, C, H, W)
        ).mean(dim=(1, 2, 3))  # Media su (C,H,W) → (B,)
        
        mse_loss = mse_per_sample.mean()  # Media su batch → scalare
        
        cos_map = F.cosine_similarity(student_norm, teacher_norm, dim=1)  # (B, H, W)
        cos_sim_per_image = cos_map.flatten(1).mean(dim=1)  # Media su (H,W) → (B,)
        cos_sim = cos_sim_per_image.mean()  # Media su batch → scalare
        
        cos_loss = 1.0 - cos_sim
        
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
    pil_images: Optional[List] = None,
) -> torch.Tensor:
    """
    Forward pass unificato per MapAnything distillation.
    Supporta sia single-view batch-safe che multi-view con cross-attention.
    Supporta anche l'iniezione diretta di immagini PIL (per on-the-fly augmentation).
    
    Args:
        model: MapAnything model
        image_paths: Lista di path immagini (flat list)
        device: Device CUDA
        use_amp: Se usare mixed precision
        amp_dtype: Tipo AMP ("bf16" o "fp16")
        process_individually: Se True, ogni immagine è processata separatamente.
        pil_images: Lista opzionale di immagini PIL (già aumentate). Se fornita, usa queste invece di caricare da disco.
    
    Returns:
        torch.Tensor: Student features (B, C, H, W) dove B = len(image_paths)
    """
    from mapanything.utils.image import load_images
    
    amp_dtype_torch = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    # Normalization constants per MapAnything (ImageNet stats)
    # Necessario quando convertiamo manualmente da PIL a Tensor
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def prepare_views_from_pil(pils, paths):
        """Helper per convertire PIL -> Tensor normalizzato -> Dict format per MapAnything"""
        views_list = []
        for i, img in enumerate(pils):
            # Convert PIL to Tensor (0-1)
            # transforms.functional.to_tensor converte HWC [0,255] -> CHW [0,1]
            tensor_img = transforms.functional.to_tensor(img).to(device).unsqueeze(0) # Aggiungi batch dim (1, C, H, W)
            
            # Normalize
            tensor_img = (tensor_img - mean) / std
            
            # Rimuovi batch dim per il formato atteso (C, H, W) dentro la lista views
            tensor_img = tensor_img.squeeze(0)

            views_list.append({
                "img": tensor_img,
                "path": paths[i] if paths else f"dummy_{i}",
                # MapAnything gestirà K=None con i default o intrinsics stimate se necessario
                # Nota: Se facessimo crop geometrici in MV, qui dovremmo aggiornare K
                "K": None, 
            })
        return views_list

    # --- RAMO 1: IMMAGINI AUGMENTED (IN MEMORIA) ---
    if pil_images is not None:
        if process_individually:
            # Single-view: processa una alla volta per evitare cross-attention
            all_features = []
            for i, pil_img in enumerate(pil_images):
                # Prepara singola view
                views = prepare_views_from_pil([pil_img], [image_paths[i]])
                
                with torch.autocast("cuda", enabled=use_amp, dtype=amp_dtype_torch):
                    _ = model(views, memory_efficient_inference=False)
                
                base_model = model.module if hasattr(model, "module") else model
                student_features_single = getattr(base_model, "_last_feat2_8x", None)
                if student_features_single is None:
                    raise KeyError("Student features not found (_last_feat2_8x).")
                
                all_features.append(student_features_single)
            
            return torch.cat(all_features, dim=0)
        
        else:
            # Multi-view: processa tutte le immagini insieme (attiva cross-attention)
            views = prepare_views_from_pil(pil_images, image_paths)
            
            with torch.autocast("cuda", enabled=use_amp, dtype=amp_dtype_torch):
                _ = model(views, memory_efficient_inference=False)
            
            base_model = model.module if hasattr(model, "module") else model
            student_features = getattr(base_model, "_last_feat2_8x", None)
            if student_features is None:
                raise KeyError("Student features not found (_last_feat2_8x).")
            
            return student_features

    # --- RAMO 2: CARICAMENTO DA DISCO (LEGACY / PRECOMPUTED) ---
    else:
        if process_individually:
            # Single-view batch-safe
            all_features = []
            for img_path in image_paths:
                views = load_images([img_path])
                
                with torch.autocast("cuda", enabled=use_amp, dtype=amp_dtype_torch):
                    for v in views:
                        img = v.get("img")
                        if isinstance(img, torch.Tensor):
                            v["img"] = img.to(device, non_blocking=True)
                    
                    _ = model(views, memory_efficient_inference=False)
                
                base_model = model.module if hasattr(model, "module") else model
                student_features_single = getattr(base_model, "_last_feat2_8x", None)
                if student_features_single is None:
                    raise KeyError("Student features not found (_last_feat2_8x).")
                
                all_features.append(student_features_single)
            
            return torch.cat(all_features, dim=0)
        
        else:
            # Multi-view
            views = load_images(image_paths)
            
            with torch.autocast("cuda", enabled=use_amp, dtype=amp_dtype_torch):
                for v in views:
                    img = v.get("img")
                    if isinstance(img, torch.Tensor):
                        v["img"] = img.to(device, non_blocking=True)
                
                _ = model(views, memory_efficient_inference=False)
            
            base_model = model.module if hasattr(model, "module") else model
            student_features = getattr(base_model, "_last_feat2_8x", None)
            if student_features is None:
                raise KeyError("Student features not found (_last_feat2_8x).")
            
            return student_features

def train_one_epoch_distillation(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    teacher_extractor: Optional[TeacherFeatureExtractor] = None,
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

        # [DEBUG] Stampa scene e views (raggruppate per cartella padre)
        # if args.multi_view_mode:
        #     try:
        #         by_scene = {}
        #         for p in image_paths:
        #             scene = str(Path(p).parent)
        #             by_scene.setdefault(scene, []).append(p)
        #         print(f"[Train][SceneViews] Batch {data_iter_step}:")
        #         for scene, views in sorted(by_scene.items()):
        #             print(f"  {scene}")
        #             for v in views:
        #                 print(f"    - {v}")
        #     except Exception:
        #         pass
        
        # Extract or load teacher features
        if mode == "precomputed":
            teacher_features = batch["teacher_features"].to(device, non_blocking=True)
        else:  # online
            if teacher_extractor is None:
                raise ValueError("teacher_extractor required for online mode but not provided")
            
            pil_images = batch["pil_images"]
            with torch.no_grad():
                teacher_features = teacher_extractor(pil_images).to(device, non_blocking=True)
        
        pil_images = batch.get("pil_images")

        # Forward pass to get student features
        student_features = forward_pass_distillation_unified(
            model=model,
            image_paths=image_paths,
            device=device,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
            process_individually=not args.multi_view_mode,
            pil_images=pil_images if mode == "online" else None
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

        # Compute loss
        loss, loss_details = criterion(student_features, teacher_features)
        mse_value = float(loss_details.get("mse_loss", 0.0))
        cos_value = float(loss_details.get("cos_loss", 0.0))
        cos_sim_value = float(loss_details.get("cos_sim", 0.0))

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
            optimizer.zero_grad()

        # Accumulate weighted sums
        batch_size = student_features.shape[0]
        total_samples += batch_size
        sum_loss += loss_value * batch_size
        sum_mse += mse_value * batch_size
        sum_cos += cos_value * batch_size
        sum_cos_sim += cos_sim_value * batch_size
        sum_mean_diff += md * batch_size
        sum_std_diff += sd * batch_size

        # Clean up
        del loss, student_features, teacher_features
        
        # Update metrics
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss_value, **loss_details)

    # Return averaged stats
    denom = max(1, total_samples)
    results = {
        "loss_mean": sum_loss / denom,
        "mse_loss_mean": sum_mse / denom,
        "cos_loss_mean": sum_cos / denom,
        "cos_sim_mean": sum_cos_sim / denom,
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
    teacher_extractor: Optional[TeacherFeatureExtractor] = None,  # ← NEW
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

        # [DEBUG] Stampa scene e views (raggruppate per cartella padre)
        # if args.multi_view_mode:
        #     try:
        #         by_scene = {}
        #         for p in image_paths:
        #             scene = str(Path(p).parent)
        #             by_scene.setdefault(scene, []).append(p)
        #         print(f"[Val][SceneViews] Batch {batch_idx}:")
        #         for scene, views in sorted(by_scene.items()):
        #             print(f"  {scene}")
        #             for v in views:
        #                 print(f"    - {v}")
        #     except Exception:
        #         pass
        
        # Extract or load teacher features
        if mode == "precomputed":
            teacher_features = batch["teacher_features"].to(device, non_blocking=True)
        else:
            pil_images = batch["pil_images"]
            teacher_features = teacher_extractor(pil_images).to(device, non_blocking=True)

        pil_images = batch.get("pil_images")
        
        # Forward pass
        student_features = forward_pass_distillation_unified(
            model=model,
            image_paths=image_paths,
            device=device,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
            process_individually=not args.multi_view_mode,
            pil_images=pil_images if mode == "online" else None
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
        
        # Compute loss
        loss, loss_details = criterion(student_features, teacher_features)
        loss_value = loss.detach().cpu().item()
        mse_value = float(loss_details.get("mse_loss", 0.0))
        cos_value = float(loss_details.get("cos_loss", 0.0))
        cos_sim_value = float(loss_details.get("cos_sim", 0.0))

        try:
            md, sd, cs = mean_std_difference(student_features, teacher_features)
            md = float(md)
            sd = float(sd)
            cs = float(cs)
        except Exception:
            md = sd = 0.0
            cs = cos_sim_value

        metric_logger.update(loss=loss_value, mse_loss=mse_value, cos_loss=cos_value, cos_sim=cos_sim_value)
        
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

        del student_features, teacher_features

    denom = max(1, total_samples)
    results = {
        "loss_mean": sum_loss / denom,
        "mse_loss_mean": sum_mse / denom,
        "cos_loss_mean": sum_cos / denom,
        "cos_sim_mean": sum_cos_sim / denom,
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
        teacher_extractor = TeacherFeatureExtractor(
            checkpoint_path=SAM2_PATH,
            device=str(device),
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
        no_augmentation=args.no_augmentation,
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
        no_augmentation=True, # Validazione sempre senza random augmentation (solo resize)
    )

    # Logica Debug per MULTI-VIEW
    if args.multi_view_mode and args.debug_max_val_images:
        original_len = len(data_loader_val.dataset.samples)
        limit = min(args.debug_max_val_images, original_len)
        data_loader_val.dataset.samples = data_loader_val.dataset.samples[:limit]
        print(f"[DEBUG] Multi-View: Limited val to first {limit} SCENES (was {original_len})")
    
    # ========== LOAD MODEL ==========
    print("Loading MapAnything model...")
    if global_rank == 0:
        model = MapAnything.from_pretrained(args.model_name, strict=False).to(device)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if global_rank != 0:
        model = MapAnything.from_pretrained(args.model_name, strict=False).to(device)
    
    model_without_ddp = model
    print(f"Model loaded. Has dpt_feature_head_2: {hasattr(model, 'dpt_feature_head_2')}")

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
            args.info_sharing_unfrozen_indices = unfrozen_indices
            print(f"[INFO] Unfroze last {num_info_sharing_blocks} info_sharing blocks (indices {unfrozen_indices})")
            print(f"[INFO] Unfroze {unfrozen_count:,} parameters in info_sharing")
        else:
            args.info_sharing_unfrozen_indices = []
    else:
        args.info_sharing_unfrozen_indices = []

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
    
    # Initialize criterion
    criterion = DistillationLoss(
        mse_weight=args.mse_weight,
        cosine_weight=args.cosine_weight,
        normalize=args.normalize_features,
    ).to(device)

    # ========== OPTIMIZER con LR differenziati ==========
    head_params = []
    encoder_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("dpt_feature_head_2"):
            head_params.append(p)
        elif name.startswith("info_sharing") or name.startswith("sam2_compat"):
            encoder_params.append(p)
        else:
            other_params.append(p)

    # Fallback: se alcuni parametri trainabili non rientrano nelle categorie, mettili nel gruppo head
    if other_params:
        print(f"[WARN] {sum(op.numel() for op in other_params):,} trainable params not matched; assigning to HEAD LR group.")
        head_params.extend(other_params)

    lr_head = args.lr
    lr_encoder = args.lr * args.lr_encoder_scale

    optimizer = optim.AdamW(
        [
            {"params": head_params, "lr": lr_head},
            {"params": encoder_params, "lr": lr_encoder},
        ],
        lr=args.lr,  # non usato per i gruppi espliciti, rimane come default
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    print(f"[OPT] Groups: head={sum(p.numel() for p in head_params):,} params @ LR {lr_head}, "
          f"encoder={sum(p.numel() for p in encoder_params):,} params @ LR {lr_encoder}")

    # # ========== OPTIMIZER ==========
    # trainable_params = [p for p in model.parameters() if p.requires_grad]

    # optimizer = optim.AdamW(
    #     trainable_params,
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    #     betas=(0.9, 0.95),
    # )
    # print(optimizer)

    #################### DEBUG ########################
    # # subito DOPO aver creato optimizer (una sola volta)
    # print("OPTIMIZER param groups summary:")
    # for i, pg in enumerate(optimizer.param_groups):
    #     n_params = sum(p.numel() for p in pg['params'])
    #     print(f" pg {i}: lr={pg.get('lr')}, params={n_params:,}")
    # # totale trainable come prima stampata
    # trainable_opt_params = sum(p.numel() for p in optimizer.param_groups[0]['params'])
    # print("Total params in first group (approx):", trainable_opt_params)
    ###################################################
    
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
            step_size=args.lr_decay_steps,
            gamma=0.1,
        )
        print(f"[INFO] Using StepLR with step_size={args.lr_decay_steps}, gamma=0.1")
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

        optimizer.load_state_dict(ckpt["optimizer"])

        if args.lr_scheduler == "none" or args.override_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            print(f"[INFO] Overriding optimizer LR to {args.lr}")

        # Scheduler resume logic with T_max override
        if args.lr_scheduler != "none" and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            # If user provided a new T_max, overwrite it in the scheduler
            if hasattr(scheduler, "T_max") and getattr(args, "overwrite_scheduler_t_max", False):
                old_tmax = getattr(scheduler, "T_max", None)
                scheduler.T_max = args.lr_scheduler_t_max
                print(f"[INFO] Overriding scheduler T_max: {old_tmax} -> {scheduler.T_max}")
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
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--accum_iter", type=int, default=1, help="Gradient accumulation iterations")
    
    # Learning rate and scheduler
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate") # Usa 5e-4 per BS_eff = 16 --> 1e-3 per BS_eff = 32, 2.5e-4 per BS_eff = 8
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum learning rate for scheduler")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine","step", "plateau", "none"])
    parser.add_argument("--plateau_patience", type=int, default=10, help="Patience for ReduceLROnPlateau scheduler")
    parser.add_argument("--lr_decay_steps", type=int, default=1000, help="Steps per decay x0.1 (StepLR)")
    parser.add_argument("--lr_scheduler_t_max", type=int, default=None, help="T_max for CosineAnnealingLR")
    parser.add_argument("--override_lr", action="store_true", help="Override LR from checkpoint with --lr value")
    parser.add_argument("--overwrite_scheduler_t_max", action="store_true", help="Overwrite scheduler T_max when resuming")
    
    # Mixed precision
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="AMP dtype")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--disable_cudnn_benchmark", action="store_true", help="Disable cudnn benchmark")

    # Loss
    parser.add_argument("--mse_weight", type=float, default=0.5, help="Weight for MSE loss")
    parser.add_argument("--cosine_weight", type=float, default=0.5, help="Weight for cosine loss")
    parser.add_argument("--normalize_features", action="store_true", help="Normalize features before loss")
    
    # Data
    parser.add_argument("--dataset", type=str, default="coco2017", choices=["coco2017", "ETH3D", "ETH3D_single"], help="Seleziona il dataset")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--debug_max_train_images", type=int, default=None, help="Limit training images for debugging")
    parser.add_argument("--debug_max_val_images", type=int, default=None, help="Limit validation images for debugging")
    parser.add_argument("--precomputed_features", action="store_true", help="Use precomputed features from disk")
    
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
    parser.add_argument("--num_info_sharing_blocks_unfreeze", type=int, default=0, help="Number of last info_sharing transformer blocks to unfreeze")
    parser.add_argument("--lr_encoder_scale", type=float, default=0.1, help="Scale factor for encoder LR relative to --lr")
    
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
