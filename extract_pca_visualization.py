#!/usr/bin/env python3
"""
Script essenziale per estrarre e visualizzare features PCA da MapAnything.

Uso:
    python extract_pca_visualization.py \
        --checkpoint_path /path/to/checkpoint.pth \
        --image_path /path/to/image.jpg \
        --output_dir /path/to/output

Output:
    Salva visualizzazione PCA in output_dir/visualizations/
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import numpy as np

# Aggiungi path per imports
sys.path.insert(0, str(Path(__file__).parent))

from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from nico.utils import create_student_original_teacher_side_by_side


def load_model_with_checkpoint(checkpoint_path: str, device: torch.device):
    """Carica il modello MapAnything e il checkpoint."""
    print(f"[INFO] Loading MapAnything model...")
    
    # Carica modello base
    model = MapAnything.from_pretrained(
            "facebook/map-anything",
            revision="562de9ff7077addd5780415661c5fb031eb8003e",
            strict=False,
            # local_files_only=True,
        ).to(device)
    
    print(f"[INFO] Model loaded. Has dpt_feature_head_2: {hasattr(model, 'dpt_feature_head_2')}")
    
    # Carica checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Estrai state_dict (gestisce vari formati di checkpoint)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Carica weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"[WARN] Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"[WARN] Unexpected keys: {unexpected_keys[:5]}...")
        
        print(f"[INFO] Checkpoint loaded successfully")
        
        # Stampa epoch se disponibile
        if "epoch" in checkpoint:
            print(f"[INFO] Checkpoint epoch: {checkpoint['epoch']}")
    else:
        print(f"[WARN] Checkpoint not found at {checkpoint_path}, using pretrained weights only")
    
    model.eval()
    return model


def extract_features(model, image_path: str, device: torch.device) -> torch.Tensor:
    """Estrae features da una singola immagine usando dpt_feature_head_2."""
    print(f"[INFO] Processing image: {image_path}")
    
    # Carica immagine
    images_dict = load_images([image_path], size=512, verbose=False)
    image_tensor = images_dict["img"]  # (1, 3, H, W)
    image_tensor = image_tensor.to(device)
    
    print(f"[INFO] Image tensor shape: {image_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(image_tensor, mode="infer")
        
        # Estrai features da dpt_feature_head_2
        if hasattr(model, 'dpt_feature_head_2'):
            features = output.get('feature_upsampled_8x', None)
            if features is None:
                raise RuntimeError("feature_upsampled_8x not found in model output")
        else:
            raise RuntimeError("Model does not have dpt_feature_head_2")
    
    print(f"[INFO] Extracted features shape: {features.shape}")
    return features


def save_visualization(
    features: torch.Tensor,
    image_path: str,
    output_dir: str,
):
    """Salva visualizzazione PCA delle features."""
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Move to CPU
    features_cpu = features.detach().cpu()  # (1, C, H, W)
    
    # Nome output basato sull'immagine
    image_name = Path(image_path).stem
    output_path = viz_dir / f"{image_name}_pca.jpg"
    
    print(f"[INFO] Creating PCA visualization...")
    
    # Usa la funzione da nico.utils (passa due volte le stesse features)
    create_student_original_teacher_side_by_side(
        student_features=features_cpu[0],  # (C, H, W)
        teacher_features=features_cpu[0],  # (C, H, W) - same as student
        original_image_path=image_path,
        output_path=str(output_path),
        title=f"Features PCA - {image_name}",
    )
    
    print(f"[INFO] Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract and visualize PCA features from MapAnything")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_pca",
        help="Output directory for visualizations (default: ./output_pca)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu, default: cuda)",
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Verifica che i file esistano
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    
    # Carica modello
    model = load_model_with_checkpoint(args.checkpoint_path, device)
    
    # Estrai features
    features = extract_features(model, args.image_path, device)
    
    # Salva visualizzazione
    save_visualization(features, args.image_path, args.output_dir)
    
    print("[INFO] Done!")


if __name__ == "__main__":
    main()