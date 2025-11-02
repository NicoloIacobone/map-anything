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

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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


# ==================== Runtime/Environment Settings ====================
# Rilevazione ambiente ed impostazione percorsi:
# - se non c'è un TTY (tipico dei job su cluster), run_cluster=True;
# - si impostano cache/percorsi di input, output e dataset COCO2017 coerenti;
# - i DataLoader useranno i path derivati dalle costanti sottostanti.
# Questo evita di dover passare i path da CLI e mantiene coerenza con distillation.py.
# Determine if we're running on the cluster (no TTY) and set paths accordingly
run_cluster = not sys.stdout.isatty()
if run_cluster:
    # Optional: set torch hub cache to a persistent location on cluster
    os.environ["TORCH_HOME"] = "/cluster/home/niacobone/torch_cache"
    try:
        import torch.hub as _torch_hub
        _torch_hub.set_dir(os.environ["TORCH_HOME"])
        print(f"[INFO] Torch hub cache dir set to {_torch_hub.get_dir()}")
    except Exception:
        pass

    INPUT_DIR = "/cluster/scratch/niacobone/distillation/training_samples"
    BASE_DIR = "/cluster/work/igp_psr/niacobone/distillation/output"
    COCO2017_ROOT = "/cluster/scratch/niacobone/distillation/coco2017"
else:
    INPUT_DIR = "/scratch2/nico/distillation/training_samples"
    BASE_DIR = "/scratch2/nico/distillation/output"
    COCO2017_ROOT = "/scratch2/nico/distillation/coco2017"

# Dataset directory structure (consistent with distillation.py)
IMAGES_DIRNAME = "val2017"
FEATURES_DIRNAME = "teacher_features"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"

TRAIN_IMAGES_DIR = os.path.join(COCO2017_ROOT, TRAIN_SPLIT, IMAGES_DIRNAME)
VAL_IMAGES_DIR = os.path.join(COCO2017_ROOT, VAL_SPLIT, IMAGES_DIRNAME)
TRAIN_FEATURES_DIR = os.path.join(COCO2017_ROOT, TRAIN_SPLIT, FEATURES_DIRNAME)
VAL_FEATURES_DIR = os.path.join(COCO2017_ROOT, VAL_SPLIT, FEATURES_DIRNAME)


# ==================== Dataset Classes ====================

class DistillationDataset(Dataset):
    """
    Dataset per la distillazione: carica immagini e le corrispondenti feature del teacher.
    
    Args:
        image_dir: cartella contenente le immagini.
        features_dir: cartella con le feature del teacher pre-computate (.pt per immagine),
            salvate come <basename>.pt.
        image_paths: lista opzionale di path da usare; se None, scansiona image_dir.
        transform: eventuale trasformazione (non usata direttamente; il caricamento
            vero per il modello avviene con load_images nella forward stage).
    """
    
    def __init__(
        self,
        image_dir: str,
        features_dir: str,
        image_paths: Optional[List[str]] = None,
        transform=None,
    ):
        self.image_dir = Path(image_dir)
        self.features_dir = Path(features_dir)
        self.transform = transform
        
        # Scan for images if not provided
        if image_paths is None:
            self.image_paths = sorted([
                str(self.image_dir / f)
                for f in os.listdir(self.image_dir)
                if self._is_image_file(f)
            ])
        else:
            self.image_paths = image_paths
            
        print(f"[Dataset] Loaded {len(self.image_paths)} images from {image_dir}")
    
    @staticmethod
    def _is_image_file(name: str) -> bool:
        """Check if file is an image based on extension."""
        return name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Ritorna:
            - 'image_path': path all'immagine
            - 'teacher_features': Tensor (C,H,W) o (1,C,H,W) con le feature del teacher
        La corrispondenza immagine→file di feature avviene usando lo stem del filename.
        """
        img_path = self.image_paths[idx]
        img_name = Path(img_path).stem
        
        # Carica le feature del teacher dal file <image_stem>.pt nella cartella features_dir
        feat_path = self.features_dir / f"{img_name}.pt"
        if not feat_path.exists():
            raise FileNotFoundError(f"Teacher features not found: {feat_path}")
        
        teacher_feat = torch.load(feat_path, map_location="cpu")
        
        # Se salvate con batch dim=1, rimuove la dimensione per ottenere (C,H,W)
        if teacher_feat.ndim == 4 and teacher_feat.shape[0] == 1:
            teacher_feat = teacher_feat.squeeze(0)
        
        return {
            "image_path": img_path,
            "teacher_features": teacher_feat,
        }


def collate_fn_distillation(batch: List[Dict]) -> Dict:
    """
    Collate function personalizzata per il dataset di distillazione.
    
    Returns:
        Dict con:
            - 'image_paths': lista di path (B)
            - 'teacher_features': Tensor (B,C,H,W) ottenuto dallo stack delle feature
    """
    image_paths = [item["image_path"] for item in batch]
    teacher_feats = torch.stack([item["teacher_features"] for item in batch], dim=0)
    
    return {
        "image_paths": image_paths,
        "teacher_features": teacher_feats,
    }


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
        
        # MSE loss
        mse_loss = F.mse_loss(student_norm, teacher_norm)
        
        # Cosine similarity loss (1 - cosine_similarity)
        cos_sim = F.cosine_similarity(student_norm, teacher_norm, dim=1).mean()
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
    features_dir: str,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    image_paths: Optional[List[str]] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for distillation training/validation.
    
    Args:
        image_dir: Directory containing images
        features_dir: Directory containing teacher features
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the dataset
        image_paths: Optional list of specific image paths to use
        pin_memory: Whether to use pinned memory
    
    Returns:
        DataLoader per distillazione
    """
    dataset = DistillationDataset(
        image_dir=image_dir,
        features_dir=features_dir,
        image_paths=image_paths,
    )
    
    # drop_last viene messo a True quando shuffle=True (training) per evitare batch finali incompleti
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_distillation,
        drop_last=shuffle,  # Drop last incomplete batch during training
    )
    
    return loader


# ==================== Training Functions ====================

def forward_pass_distillation(
    model: torch.nn.Module,
    image_paths: List[str],
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: str = "bf16",
) -> torch.Tensor:
    """
    Esegue la forward di MapAnything per estrarre le feature dello studente
    dalla testa dpt_feature_head_2 (key '_last_feat2_8x').
    
    Args:
        model: MapAnything model
        image_paths: List of image paths for this batch
        device: Device to run on
        use_amp: Whether to use automatic mixed precision
        amp_dtype: AMP dtype ("bf16" or "fp16")
    
    Returns:
        student_features: (B, C, H, W) tensor of student features from dpt_feature_head_2
    """
    # Carica le immagini usando l'utility del progetto (gestisce pre-processing coerente)
    views = load_images(image_paths)

    # Sposta i tensori immagine sul device per evitare mismatch CPU/GPU (come in distillation.py)
    for v in views:
        img = v.get("img")
        if isinstance(img, torch.Tensor):
            v["img"] = img.to(device, non_blocking=True)
    
    # Determine autocast dtype
    if amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16

    # Abilita autocast solo se siamo su CUDA
    autocast_enabled = use_amp and (device.type == "cuda")
    
    # Forward pass con autocast (AMP); si usa forward (non infer) per mantenere gradienti
    with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=autocast_dtype):
        predictions = model(
            views,
            memory_efficient_inference=False,
        )
    
    # Estrai le feature dello studente dall'attributo popolato nel forward del modello
    # (come fa distillation.py): atteso 'model._last_feat2_8x' con shape (B,C,H,W)
    student_features = getattr(model, "_last_feat2_8x", None)
    if student_features is None:
        raise KeyError(
            "Student features not found on model (_last_feat2_8x). "
            "Ensure dpt_feature_head_2 populates model._last_feat2_8x during forward."
        )
    
    return student_features


def train_one_epoch_distillation(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
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
    
    Returns:
        Dictionary of averaged training metrics
    """
    model.train(True)
    metric_logger = train_tools.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", train_tools.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Distillation Epoch: [{epoch}]"
    
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    # Accumulatori per medie pesate sull'intera epoca (coerenti con distillation.py):
    # sommiamo loss/metriche pesate per batch_size e dividiamo a fine epoca.
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
        epoch_f = epoch + data_iter_step / max(1, len(data_loader))
        
        # Get data
        image_paths = batch["image_paths"]
        teacher_features = batch["teacher_features"].to(device, non_blocking=True)
        
        # Forward pass to get student features
        student_features = forward_pass_distillation(
            model=model,
            image_paths=image_paths,
            device=device,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
        )
        
        # Resize teacher features to match student resolution if needed
        if teacher_features.shape[-2:] != student_features.shape[-2:]:
            H, W = student_features.shape[-2:]
            teacher_features = F.interpolate(
                teacher_features,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
        
        # Compute loss
        loss, loss_details = criterion(student_features, teacher_features)
        loss_value = float(loss)
        mse_value = float(loss_details.get("mse_loss", 0.0))
        cos_value = float(loss_details.get("cos_loss", 0.0))
        cos_sim_value = float(loss_details.get("cos_sim", 0.0))

        # Compute additional metrics to mirror distillation.py
        try:
            md, sd, cs = mean_std_difference(student_features, teacher_features)
            md = float(md)
            sd = float(sd)
            cs = float(cs)
        except Exception:
            md = sd = 0.0
            cs = cos_sim_value
        
        # Controllo stabilità numerica: interrompe il training in caso di NaN/Inf
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training", force=True)
            print(f"Loss Details: {loss_details}", force=True)
            sys.exit(1)
        
        # Gradient Accumulation: scala la loss per accumulare su 'accum_iter' iterazioni
        loss = loss / accum_iter
        
        # Backward pass
        loss.backward()
        
        # Step ottimizzatore ogni 'accum_iter' iterazioni (simula batch più grande)
        if (data_iter_step + 1) % accum_iter == 0:
            # Gradient clipping
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
        
    # Return averaged stats (weighted by batch size)
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
    
    Returns:
        Dictionary of validation metrics (avg and median)
    """
    model.eval()
    metric_logger = train_tools.MetricLogger(delimiter="  ")
    # Finestra molto grande per rendere la stampa simile a una media globale
    metric_logger.meters = defaultdict(lambda: train_tools.SmoothedValue(window_size=9**9))
    header = f"Distillation Validation: [{epoch}]"
    
    # Manual accumulators for weighted epoch averages
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
        # Get data
        image_paths = batch["image_paths"]
        teacher_features = batch["teacher_features"].to(device, non_blocking=True)
        
        # Forward pass
        student_features = forward_pass_distillation(
            model=model,
            image_paths=image_paths,
            device=device,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
        )
        
        # Resize teacher to match student if needed
        if teacher_features.shape[-2:] != student_features.shape[-2:]:
            H, W = student_features.shape[-2:]
            teacher_features = F.interpolate(
                teacher_features,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
        
        # Compute loss
        loss, loss_details = criterion(student_features, teacher_features)
        loss_value = float(loss)
        mse_value = float(loss_details.get("mse_loss", 0.0))
        cos_value = float(loss_details.get("cos_loss", 0.0))
        cos_sim_value = float(loss_details.get("cos_sim", 0.0))

        # Additional metrics
        try:
            md, sd, cs = mean_std_difference(student_features, teacher_features)
            md = float(md)
            sd = float(sd)
            cs = float(cs)
        except Exception:
            md = sd = 0.0
            cs = cos_sim_value

        # Update metrics
        metric_logger.update(loss=loss_value, mse_loss=mse_value, cos_loss=cos_value, cos_sim=cos_sim_value)
        
        # Salva visualizzazioni se richiesto (solo primo batch per epoca per limitare I/O)
        if args.save_visualizations and batch_idx == 0:
            save_pca_visualizations(
                student_features=student_features,
                teacher_features=teacher_features,
                image_paths=image_paths,
                epoch=epoch,
                output_dir=args.output_dir,
            )
        
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
        del student_features, teacher_features
    
    # Compute aggregates (weighted means)
    denom = max(1, total_samples)
    results = {
        "loss_mean": sum_loss / denom,
        "mse_loss_mean": sum_mse / denom,
        "cos_loss_mean": sum_cos / denom,
        "cos_sim_mean": sum_cos_sim / denom,
        "mean_diff": sum_mean_diff / denom,
        "std_diff": sum_std_diff / denom,
        "samples": total_samples,
        # Backward compatibility key used elsewhere
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
        except Exception as e:
            print(f"[WARN] Failed to create PCA visualization for {img_path}: {e}")
            continue
    
    print(f"[VIZ] Saved {B} PCA visualizations to {viz_dir}")


# ==================== Main Training Loop ====================

def distill(args):
    """
    Main distillation training function.
    
    This orchestrates:
    - Dataset/DataLoader setup
    - Model initialization and freezing
    - Optimizer and scheduler setup
    - Training/validation loop
    - Checkpointing and logging
    
    Args:
        args: Configuration namespace with all hyperparameters
    """
    # Inizializza (eventualmente) il training distribuito e ricava il rank
    train_tools.init_distributed_mode(args.distributed)
    global_rank = train_tools.get_rank()
    
    # Imposta la cartella di output: se non fornita, costruisce BASE_DIR/<wandb_name|timestamp>
    if not args.output_dir:
        # Derive default output_dir from BASE_DIR and run name (prefer wandb_name)
        default_run_name = args.wandb_name or datetime.datetime.now().strftime("distill_%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(BASE_DIR, default_run_name)
    print(f"output_dir: {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Seed e impostazioni cuDNN: per riproducibilità e performance
    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = not args.disable_cudnn_benchmark
    
    # Su cluster riduci la verbosità delle stampe per contenere i log
    if run_cluster and getattr(args, "print_freq", 10) < 200:
        args.print_freq = 200

    # Costruzione DataLoader: usa path derivati da COCO2017_ROOT (come distillation.py)
    print(f"Building train dataloader from {TRAIN_IMAGES_DIR}")
    train_image_paths = None
    if args.debug_max_train_images:
        # Sottoinsieme random di immagini per debug veloce
        all_imgs = sorted([
            os.path.join(TRAIN_IMAGES_DIR, f)
            for f in os.listdir(TRAIN_IMAGES_DIR)
            if DistillationDataset._is_image_file(f)
        ])
        train_image_paths = random.sample(all_imgs, min(args.debug_max_train_images, len(all_imgs)))
    
    data_loader_train = build_distillation_dataloader(
        image_dir=TRAIN_IMAGES_DIR,
        features_dir=TRAIN_FEATURES_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        image_paths=train_image_paths,
    )
    
    print(f"Building val dataloader from {VAL_IMAGES_DIR}")
    val_image_paths = None
    if args.debug_max_val_images:
        # Primi N elementi per validazione rapida di debug
        all_val_imgs = sorted([
            os.path.join(VAL_IMAGES_DIR, f)
            for f in os.listdir(VAL_IMAGES_DIR)
            if DistillationDataset._is_image_file(f)
        ])
        val_image_paths = all_val_imgs[:args.debug_max_val_images]
    
    data_loader_val = build_distillation_dataloader(
        image_dir=VAL_IMAGES_DIR,
        features_dir=VAL_FEATURES_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        image_paths=val_image_paths,
    )
    
    # Carica il modello pre-addestrato (strict=False per permettere head extra)
    print("Loading MapAnything model...")
    if global_rank == 0:
        model = MapAnything.from_pretrained(args.model_name, strict=False).to(device)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # sincronizzazione tra processi
    if global_rank != 0:
        model = MapAnything.from_pretrained(args.model_name, strict=False).to(device)
    
    model_without_ddp = model
    print(f"Model loaded. Has dpt_feature_head_2: {hasattr(model, 'dpt_feature_head_2')}")
    
    # Congela tutto tranne la testa dpt_feature_head_2 (oggetto della distillazione)
    print("Freezing all parameters except dpt_feature_head_2...")
    for name, param in model.named_parameters():
        if not name.startswith("dpt_feature_head_2"):
            param.requires_grad = False
        # else:
        #     print(f"  Trainable: {name} | {param.shape}")
    
    # Initialize criterion
    criterion = DistillationLoss(
        mse_weight=args.mse_weight,
        cosine_weight=args.cosine_weight,
        normalize=args.normalize_features,
    ).to(device)
    
    # Wrapping in DDP se distribuito (usa la GPU locale in device_ids)
    if args.distributed.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.distributed.gpu],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module
    
    # Ottimizzatore su soli parametri addestrabili; AdamW con betas impostati come nello script originale
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    print(optimizer)
    
    # Scheduler LR: Cosine annealing per epoca, coerente con distillation.py
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.lr_scheduler_t_max,
        eta_min=args.lr_min,
    )
    
    # Resume: ricarica head 2 + optimizer + scheduler; riparte dall'epoca successiva
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume_ckpt:
        print(f"Resuming from checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        model_without_ddp.dpt_feature_head_2.load_state_dict(ckpt["dpt_feature_head_2"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
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
    
    # Ciclo di training principale: train → validazione (ogni eval_freq) → step scheduler → checkpoint → log
    print(f"Start distillation training for {args.epochs} epochs from epoch {start_epoch}")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train one epoch
        train_stats = train_one_epoch_distillation(
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
                args=args,
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
                    )
        
        # Step scheduler
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
                log_dict.update({
                    "val_loss": val_stats.get("loss_mean", 0.0),
                    "val_mse_loss": val_stats.get("mse_loss_mean", 0.0),
                    "val_cos_loss": val_stats.get("cos_loss_mean", 0.0),
                    "val_mean_diff": val_stats.get("mean_diff", 0.0),
                    "val_std_diff": val_stats.get("std_diff", 0.0),
                    "val_cosine_similarity": val_stats.get("cos_sim_mean", 0.0),
                })
            wandb.log(log_dict, step=epoch + 1)
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
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    
    # Save wandb run_id if available
    if WANDB_AVAILABLE and wandb.run is not None:
        state["wandb_run_id"] = wandb.run.id
    
    ckpt_path = Path(output_dir) / f"checkpoint_{tag}.pth"
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
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints and logs (default: BASE_DIR/wandb_name or timestamp)")
    # Note: dataset paths are derived from COCO2017_ROOT constants depending on run_cluster
    
    # Model
    parser.add_argument("--model_name", type=str, default="facebook/map-anything", help="MapAnything model name or path")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Minimum learning rate for scheduler")
    parser.add_argument("--lr_scheduler_t_max", type=int, default=50, help="T_max for CosineAnnealingLR")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--accum_iter", type=int, default=1, help="Gradient accumulation iterations")
    
    # Loss
    parser.add_argument("--mse_weight", type=float, default=0.5, help="Weight for MSE loss")
    parser.add_argument("--cosine_weight", type=float, default=0.5, help="Weight for cosine loss")
    parser.add_argument("--normalize_features", action="store_true", help="Normalize features before loss")
    
    # Data
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--debug_max_train_images", type=int, default=None, help="Limit training images for debugging")
    parser.add_argument("--debug_max_val_images", type=int, default=None, help="Limit validation images for debugging")
    
    # Mixed precision
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="AMP dtype")
    
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
    
    # Other
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--disable_cudnn_benchmark", action="store_true", help="Disable cudnn benchmark")
    
    # Distributed (opzionale): abilita DDP; dist_url di solito 'env://' con torchrun; local_rank impostato da torchrun
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--dist_url", type=str, default="env://", help="URL for distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    
    return parser


# ==================== Entry Point ====================

def main():
    """
    Entry point for distillation training script.
    """
    parser = get_args_parser()
    args = parser.parse_args()
    
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
