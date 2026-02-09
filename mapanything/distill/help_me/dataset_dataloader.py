from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os
from PIL import Image
import torch
from torchvision import transforms
import random
import datetime
import matplotlib.pyplot as plt
import numpy as np

from sam2_minimal.sam2_builder import (
    load_sam2_feature_extractor,
)

# ==================== Dataset Classes ====================
class DistillationDataset(Dataset):
    """
    Dataset per la distillazione: supporta Single-View (lista piatta)
    """
    
    def __init__(
        self,
        image_dir: str,
        teacher_extractor: Optional[callable] = None,
        image_paths: Optional[List[str]] = None,
        transform=None,
        split: str = "train",
    ):
        self.image_dir = Path(image_dir)
        self.teacher_extractor = teacher_extractor
        self.transform = transform
        self.is_train = "train" in split.lower()
        
        # Validation
        if self.teacher_extractor is None:
            raise ValueError("Teacher_extractor must be provided")
        
        # Discovery dei samples
        self.samples = [] 
        if image_paths is not None:
            # Se paths forniti manualmente (es. debug), usiamo quelli.
            self.samples = image_paths
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
        """
        sample = self.samples[idx] # Può essere str (single) o List[str] (multi)
        
        # Normalizziamo tutto a liste per gestire single/multi uniformemente qui dentro
        img_paths = sample if isinstance(sample, list) else [sample]
        
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
        # self.augment_cfg = augment_cfg or {}
        # self._build_augment_pipelines()
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
    def __call__(self, pil_images: List, debug_visualize: bool = False) -> torch.Tensor:
        """Estrae feature con augmentation e opzionale debug visualizzazione."""
        
        features = []
        # use_aug = self.augment_cfg.get("enabled", False)
        # print(f"[INFO] Extracting features with augmentation: {use_aug}")
        
        # # DEBUG: Salva PRIMA augmentation
        # if debug_visualize and use_aug:
        #     debug_dir = Path("debug_augmentation")
        #     debug_dir.mkdir(exist_ok=True)
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
        #     for idx, pil_img in enumerate(pil_images):
        #         pil_img.save(debug_dir / f"{timestamp}_before_{idx:02d}.png")
        
        # # Applica augmentation
        # if use_aug and self.augment_shared is not None:
        #     scene_seed = random.randint(0, 2**31-1)
        #     augmented_imgs = []
        #     for pil_img in pil_images:
        #         torch.manual_seed(scene_seed)
        #         random.seed(scene_seed)
        #         augmented_imgs.append(self.augment_shared(pil_img))
        # else:
        #     augmented_imgs = []
        #     for pil_img in pil_images:
        #         if use_aug and self.augment_single is not None:
        #             augmented_imgs.append(self.augment_single(pil_img))
        #         else:
        #             augmented_imgs.append(pil_img)
        
        # # DEBUG: Salva DOPO augmentation + comparison
        # if debug_visualize and use_aug:
        #     for idx, aug_img in enumerate(augmented_imgs):
        #         aug_img.save(debug_dir / f"{timestamp}_after_{idx:02d}.png")
            
        #     n = len(pil_images)
        #     fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
        #     if n == 1:
        #         axes = axes.reshape(2, 1)
            
        #     for idx in range(n):
        #         axes[0, idx].imshow(pil_images[idx])
        #         axes[0, idx].set_title(f"Before #{idx}")
        #         axes[0, idx].axis('off')
                
        #         axes[1, idx].imshow(augmented_imgs[idx])
        #         axes[1, idx].set_title(f"After #{idx}")
        #         axes[1, idx].axis('off')
            
        #     plt.tight_layout()
        #     plt.savefig(debug_dir / f"{timestamp}_comparison.png", dpi=150)
        #     plt.close()
        #     print(f"[DEBUG] Salvate immagini in {debug_dir}/")
        
        # Estrai features
        for pil_img in pil_images:
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
    split: str = "train",
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
        split=split,
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
