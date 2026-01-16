#!/usr/bin/env python3
"""
Debug script per visualizzare t_masks e s_masks salvati in file .pt

Utilizzo:
    python debug_masks.py --mask_dir /path/to/masks [--image_path /path/to/reference/image]

Visualizzazioni:
    1. Shape e statistiche
    2. PCA visualization (per trattare come feature)
    3. save_decoder_masks_visualization (heatmap side-by-side)
    4. convert_mask_decoder_output_to_showable + show_anns (SAM-style annotations)
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2
from PIL import Image as PILImage
from sklearn.decomposition import PCA

# Aggiungi root al path per importare moduli locali
sys.path.insert(0, str(Path(__file__).parent))

def pca_visualize_masks(
    t_masks: torch.Tensor,
    s_masks: torch.Tensor,
    output_dir: str = "debug_pca_masks",
):
    """
    Visualizza t_masks e s_masks usando PCA.
    
    Tratta i logit delle maschere come feature ad alta dimensionalità,
    proietta su 3 componenti principali e visualizza come RGB.
    
    Args:
        t_masks: (B, num_masks, H, W) teacher masks
        s_masks: (B, num_masks, H, W) student masks
        output_dir: Directory per salvare visualizzazioni
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    B, num_masks, H, W = t_masks.shape
    print(f"\n[PCA VIZ] Input shapes: t_masks {t_masks.shape}, s_masks {s_masks.shape}")
    
    # Applica sigmoid ai logit
    t_masks_prob = torch.sigmoid(t_masks).cpu().numpy()  # (B, num_masks, H, W)
    s_masks_prob = torch.sigmoid(s_masks).cpu().numpy()
    
    for batch_idx in range(B):
        # Estrai un singolo batch
        t_batch = t_masks_prob[batch_idx]  # (num_masks, H, W)
        s_batch = s_masks_prob[batch_idx]
        
        # Reshape in feature vector per pixel: (H*W, num_masks)
        t_pixels = t_batch.reshape(num_masks, -1).T  # (H*W, num_masks)
        s_pixels = s_batch.reshape(num_masks, -1).T
        
        # PCA su 3 componenti
        pca = PCA(n_components=min(3, num_masks))
        t_pca = pca.fit_transform(t_pixels)  # (H*W, 3)
        s_pca = pca.fit_transform(s_pixels)
        
        # Reshape back to image
        t_rgb = t_pca.reshape(H, W, -1)
        s_rgb = s_pca.reshape(H, W, -1)
        
        # Normalize to 0-1
        for i in range(t_rgb.shape[2]):
            t_min, t_max = t_rgb[:, :, i].min(), t_rgb[:, :, i].max()
            s_min, s_max = s_rgb[:, :, i].min(), s_rgb[:, :, i].max()
            if t_max > t_min:
                t_rgb[:, :, i] = (t_rgb[:, :, i] - t_min) / (t_max - t_min)
            if s_max > s_min:
                s_rgb[:, :, i] = (s_rgb[:, :, i] - s_min) / (s_max - s_min)
        
        # Pad a 3 canali se necessario
        if t_rgb.shape[2] < 3:
            pad_size = 3 - t_rgb.shape[2]
            t_rgb = np.pad(t_rgb, ((0, 0), (0, 0), (0, pad_size)), mode='constant')
            s_rgb = np.pad(s_rgb, ((0, 0), (0, 0), (0, pad_size)), mode='constant')
        
        # Visualizza
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(t_rgb)
        axes[0].set_title(f"Teacher Masks (PCA)")
        axes[0].axis("off")
        
        axes[1].imshow(s_rgb)
        axes[1].set_title(f"Student Masks (PCA)")
        axes[1].axis("off")
        
        plt.tight_layout()
        save_path = Path(output_dir) / f"pca_batch_{batch_idx:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"[PCA VIZ] Saved {save_path}")
        print(f"  Teacher PCA variance explained: {pca.explained_variance_ratio_}")


def save_decoder_masks_visualization_impl(
    student_masks: torch.Tensor,
    teacher_masks: torch.Tensor,
    output_dir: str = "debug_decoder_masks",
    image_path: Optional[str] = None,
):
    """
    Visualizza maschere come heatmaps side-by-side (teacher vs student).
    
    Args:
        student_masks: (B, num_masks, H, W) student mask logits
        teacher_masks: (B, num_masks, H, W) teacher mask logits
        output_dir: Directory per salvare
        image_path: Path a immagine di riferimento (opzionale)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    B, num_masks, H, W = student_masks.shape
    print(f"\n[DECODER VIZ] Input shapes: student {student_masks.shape}, teacher {teacher_masks.shape}")
    
    student_masks_cpu = torch.sigmoid(student_masks.detach().cpu())
    teacher_masks_cpu = torch.sigmoid(teacher_masks.detach().cpu())
    
    for batch_idx in range(B):
        # Carica immagine di riferimento se disponibile
        img_array = None
        if image_path and os.path.exists(image_path):
            try:
                img = PILImage.open(image_path).convert("RGB")
                img_array = np.array(img)
                # Ridimensiona a (H, W)
                if img_array.shape[:2] != (H, W):
                    img_array = cv2.resize(img_array, (W, H))
            except Exception as e:
                print(f"[WARN] Failed to load image {image_path}: {e}")
        
        # Crea figura
        n_cols = num_masks + (1 if img_array is not None else 0)
        n_rows = 2  # Teacher e Student
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 8))
        
        if num_masks == 1 and img_array is None:
            axes = axes.reshape(2, 1)
        elif n_cols == 1:
            axes = axes.reshape(2, 1)
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        col_idx = 0
        
        # Colonna 0: Immagine di riferimento
        if img_array is not None:
            for row in range(2):
                axes[row, col_idx].imshow(img_array)
                axes[row, col_idx].set_title("Original" if row == 0 else "")
                axes[row, col_idx].axis("off")
            col_idx += 1
        
        # Colonne 1+: Maschere
        for mask_idx in range(num_masks):
            teacher_mask = teacher_masks_cpu[batch_idx, mask_idx].numpy()
            student_mask = student_masks_cpu[batch_idx, mask_idx].numpy()
            
            # Teacher (riga 0)
            ax = axes[0, col_idx]
            im = ax.imshow(teacher_mask, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"T Mask {mask_idx}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Student (riga 1)
            ax = axes[1, col_idx]
            im = ax.imshow(student_mask, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"S Mask {mask_idx}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            col_idx += 1
        
        plt.tight_layout()
        save_path = Path(output_dir) / f"decoder_masks_batch_{batch_idx:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"[DECODER VIZ] Saved {save_path}")


def convert_mask_decoder_output_to_showable(
    masks_logits: torch.Tensor,
    iou_preds: torch.Tensor,
    mask_threshold: float = 0.0,
) -> dict:
    """
    Converte output del decoder SAM2 a formato showable per show_anns().
    
    Args:
        masks_logits: (B, num_masks, H, W)
        iou_preds: (B, num_masks)
        mask_threshold: Soglia per binarizzare
    
    Returns:
        annotations dict con 'masks' e 'iou_predictions'
    """
    masks = torch.sigmoid(masks_logits) > mask_threshold  # (B, num_masks, H, W)
    masks = masks.cpu().numpy()
    iou_preds = iou_preds.cpu().numpy()
    
    anns = {
        "masks": masks,
        "iou_predictions": iou_preds,
    }
    return anns


def show_anns_simple(anns: dict, output_dir: str = "debug_show_anns"):
    """
    Visualizza annotazioni in stile SAM.
    
    Args:
        anns: Dict con 'masks' (B, num_masks, H, W bool) e 'iou_predictions' (B, num_masks)
        output_dir: Directory per salvare
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    masks = anns["masks"]  # (B, num_masks, H, W)
    iou_pred = anns["iou_predictions"]  # (B, num_masks)
    
    B, num_masks, H, W = masks.shape
    print(f"\n[SHOW_ANNS VIZ] Masks shape: {masks.shape}, IoU shape: {iou_pred.shape}")
    
    for batch_idx in range(B):
        fig, axes = plt.subplots(1, num_masks + 1, figsize=(5*(num_masks+1), 5))
        
        if num_masks == 1:
            axes = [axes[0], axes[1]]
        elif not isinstance(axes, np.ndarray):
            axes = [axes]
        else:
            axes = axes.tolist()
        
        # Crea overlay background
        overlay_base = np.ones((H, W, 3), dtype=np.uint8) * 255
        
        # Colonna 0: Combina tutte le maschere
        combined_mask = np.zeros((H, W), dtype=bool)
        for mask_idx in range(num_masks):
            combined_mask |= masks[batch_idx, mask_idx]
        
        axes[0].imshow(overlay_base)
        overlay = axes[0].imshow(combined_mask, cmap="gray", alpha=0.5)
        axes[0].set_title("Combined Masks")
        axes[0].axis("off")
        
        # Colonne 1+: Maschere individuali
        for mask_idx in range(num_masks):
            mask = masks[batch_idx, mask_idx]
            iou_val = iou_pred[batch_idx, mask_idx]
            
            ax = axes[min(mask_idx + 1, len(axes)-1)]
            ax.imshow(overlay_base)
            ax.imshow(mask, cmap="Reds", alpha=0.6)
            ax.set_title(f"Mask {mask_idx}\nIoU: {iou_val:.3f}")
            ax.axis("off")
        
        plt.tight_layout()
        save_path = Path(output_dir) / f"show_anns_batch_{batch_idx:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"[SHOW_ANNS VIZ] Saved {save_path}")


def print_stats(t_masks: torch.Tensor, s_masks: torch.Tensor, student_iou: Optional[torch.Tensor] = None, teacher_iou: Optional[torch.Tensor] = None):
    """
    Stampa statistiche su maschere e IoU.
    """
    print("\n" + "="*80)
    print("MASKS STATISTICS")
    print("="*80)
    
    print(f"\n[SHAPES]")
    print(f"  t_masks: {t_masks.shape}")
    print(f"  s_masks: {s_masks.shape}")
    if student_iou is not None:
        print(f"  student_iou: {student_iou.shape}")
    if teacher_iou is not None:
        print(f"  teacher_iou: {teacher_iou.shape}")
    
    # Applica sigmoid
    t_prob = torch.sigmoid(t_masks)
    s_prob = torch.sigmoid(s_masks)
    
    print(f"\n[TEACHER MASKS (post-sigmoid)]")
    print(f"  Min: {t_prob.min():.4f}, Max: {t_prob.max():.4f}")
    print(f"  Mean: {t_prob.mean():.4f}, Std: {t_prob.std():.4f}")
    print(f"  Median: {t_prob.median():.4f}")
    
    print(f"\n[STUDENT MASKS (post-sigmoid)]")
    print(f"  Min: {s_prob.min():.4f}, Max: {s_prob.max():.4f}")
    print(f"  Mean: {s_prob.mean():.4f}, Std: {s_prob.std():.4f}")
    print(f"  Median: {s_prob.median():.4f}")
    
    # MSE e Cosine similarity
    mse = F.mse_loss(s_prob, t_prob).item()
    print(f"\n[LOSS]")
    print(f"  MSE (student vs teacher): {mse:.6f}")
    
    # Cosine similarity (flatten spaziale)
    B, num_masks, H, W = s_prob.shape
    s_flat = s_prob.reshape(B, num_masks, -1)  # (B, num_masks, H*W)
    t_flat = t_prob.reshape(B, num_masks, -1)
    
    cos_sim = F.cosine_similarity(s_flat, t_flat, dim=-1)  # (B, num_masks)
    print(f"  Cosine Similarity (mean over all): {cos_sim.mean():.6f}")
    print(f"    Per mask:")
    for m in range(num_masks):
        print(f"      Mask {m}: {cos_sim[:, m].mean():.6f}")
    
    if student_iou is not None and teacher_iou is not None:
        print(f"\n[IoU PREDICTIONS]")
        print(f"  Teacher IoU: min={teacher_iou.min():.4f}, max={teacher_iou.max():.4f}, mean={teacher_iou.mean():.4f}")
        print(f"  Student IoU: min={student_iou.min():.4f}, max={student_iou.max():.4f}, mean={student_iou.mean():.4f}")
        
        iou_mse = F.mse_loss(student_iou, teacher_iou).item()
        print(f"  IoU MSE: {iou_mse:.6f}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Debug script per visualizzare t_masks e s_masks"
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        default="/scratch2/nico/distillation/tests/masks_visualization",
        help="Directory con t_masks.pt e s_masks.pt"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="/scratch2/nico/distillation/tests/masks_visualization/reference_image.jpg",
        help="Path a immagine di riferimento (opzionale)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug_output",
        help="Directory per salvare visualizzazioni"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Esegui tutte le visualizzazioni"
    )
    
    args = parser.parse_args()
    
    # Carica tensori
    mask_dir = Path(args.mask_dir)
    t_masks_path = mask_dir / "t_masks.pt"
    s_masks_path = mask_dir / "s_masks.pt"
    
    if not t_masks_path.exists() or not s_masks_path.exists():
        print(f"[ERROR] Missing files in {mask_dir}")
        print(f"  t_masks.pt exists: {t_masks_path.exists()}")
        print(f"  s_masks.pt exists: {s_masks_path.exists()}")
        sys.exit(1)
    
    print(f"[INFO] Loading masks from {mask_dir}")
    t_masks = torch.load(t_masks_path)
    s_masks = torch.load(s_masks_path)
    
    # Carica IoU se disponibili
    student_iou = None
    teacher_iou = None
    if (mask_dir / "s_iou.pt").exists():
        student_iou = torch.load(mask_dir / "s_iou.pt")
    if (mask_dir / "t_iou.pt").exists():
        teacher_iou = torch.load(mask_dir / "t_iou.pt")
    
    # Stampa statistiche
    print_stats(t_masks, s_masks, student_iou, teacher_iou)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualizzazioni
    print("[INFO] Starting visualizations...")
    
    print("\n1. PCA Visualization")
    pca_visualize_masks(
        t_masks, s_masks,
        output_dir=str(output_dir / "pca")
    )
    
    print("\n2. Decoder Masks Visualization (heatmaps)")
    save_decoder_masks_visualization_impl(
        s_masks, t_masks,
        output_dir=str(output_dir / "decoder_masks"),
        image_path=args.image_path
    )
    
    print("\n3. Show Anns Visualization")
    anns = convert_mask_decoder_output_to_showable(s_masks, student_iou if student_iou is not None else torch.zeros(s_masks.shape[0], s_masks.shape[1]))
    show_anns_simple(
        anns,
        output_dir=str(output_dir / "show_anns")
    )
    
    print(f"\n[SUCCESS] All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()