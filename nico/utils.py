import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import torch.nn.functional as F
import shutil
import wandb
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torch import nn
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
from sam2_minimal.utils.amg import batched_mask_to_box, box_xyxy_to_xywh

class SAM2CompatibilityLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        # self.ln = nn.LayerNorm(channels, eps=1e-6)
        # self.proj = nn.Conv2d(channels, 256, kernel_size=1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ln = nn.LayerNorm(in_channels)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):  # x: (B,C,H,W)
        # LayerNorm expects (B,H*W,C) or (B,C)
        # B, C, H, W = x.shape
        # x = x.permute(0, 2, 3, 1)        # (B,H,W,C)
        # x = self.ln(x)                  # LN over C
        # x = x.permute(0, 3, 1, 2)       # (B,C,H,W)
        # x = self.proj(x)                # match SAM2 feature dim
        # return x
        b, c, h, w = x.shape
        assert c == self.in_channels, f"Expected {self.in_channels}, got {c}"
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        return x

def branch_wandb(old_id: str, new_name: str, start_epoch: int = 0) -> str:
    api = wandb.Api()
    project = "mapanything-distillation"
    old_id = "nicolo-iacobone-politecnico-di-torino/" + project + old_id
    old = api.run(old_id)

    new = wandb.init(project=project, name=new_name)
    def _flatten_metrics(obj, parent_key='', sep='/'):
        items = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.startswith('_'):
                    continue
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(_flatten_metrics(v, new_key, sep=sep))
                elif isinstance(v, (list, tuple)):
                    for i, elem in enumerate(v):
                        items.update(_flatten_metrics(elem, f"{new_key}{sep}{i}", sep=sep))
                else:
                    items[new_key] = v
        else:
            items[parent_key] = obj
        return items

    for row in old.history(pandas=False):
        if "epoch" in row and row["epoch"] <= start_epoch:
            metrics = _flatten_metrics(row)
            # ensure epoch exists at top level
            metrics.setdefault("epoch", row.get("epoch"))
            new.log(metrics)
    new.finish()
    return new.id

def resize_to_64x64(feat: torch.Tensor) -> torch.Tensor:
    # feat: (B, C, H, W)
    H, W = feat.shape[-2:]
    if (H, W) == (64, 64):
        return feat
    # area solo per downsampling; bilinear per upsampling
    mode = "area" if (H > 64 or W > 64) else "bilinear"
    return F.interpolate(feat, size=(64, 64), mode=mode, align_corners=False if mode=="bilinear" else None, antialias=True if mode=="bilinear" else False)

def split_dataset(dataset_path, images_postfix="val2017", features_postfix="teacher_features",
                  val_split=0.2, seed=0, move_instead_of_copy=False):
    """
    Esegue lo split train/val del dataset mantenendo invariati i nomi dei file.

    Assunzioni:
    - Dentro dataset_path ci sono due cartelle:
        dataset_path / images_postfix
        dataset_path / features_postfix
    - In images_postfix ci sono immagini .jpg
    - In features_postfix ci sono file .pt con lo stesso basename dell'immagine
      (es: 000001.jpg <-> 000001.pt)

    Parametri:
        dataset_path (str): percorso root del dataset.
        images_postfix (str): nome cartella immagini sorgente.
        features_postfix (str): nome cartella features sorgente.
        val_split (float): frazione di file da mettere in validation.
        seed (int): seed per shuffle.
        move_instead_of_copy (bool): se True sposta i file invece di copiarli.

    Crea (se non esistono):
        dataset_path/train/<images_postfix> , dataset_path/train/<features_postfix>
        dataset_path/val/<images_postfix>   , dataset_path/val/<features_postfix>

    I nomi dei file restano invariati.

    Ritorna:
        dict con chiavi 'train' e 'val', ognuna lista dei basenames inclusi.
    """

    root = Path(dataset_path)
    img_dir = root / images_postfix
    feat_dir = root / features_postfix

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Cartella immagini non trovata: {img_dir}")
    if not feat_dir.is_dir():
        raise FileNotFoundError(f"Cartella features non trovata: {feat_dir}")

    image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() == ".jpg"])
    pairs = []
    for img_path in image_files:
        base = img_path.stem
        feat_path = feat_dir / f"{base}.pt"
        if feat_path.is_file():
            pairs.append(base)
        else:
            print(f"[WARN] Feature mancante per {img_path.name}, saltata.")

    if not pairs:
        raise RuntimeError("Nessuna coppia immagine-feature valida trovata.")

    random.seed(seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_val = int(round(n_total * val_split))
    train_basenames = pairs[:-n_val] if n_val > 0 else pairs
    val_basenames = pairs[-n_val:] if n_val > 0 else []

    def safe_transfer(src: Path, dst: Path):
        if dst.exists():
            try:
                if dst.stat().st_size == src.stat().st_size:
                    return
            except OSError:
                pass
        dst.parent.mkdir(parents=True, exist_ok=True)
        if move_instead_of_copy:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)

    def export_split(split_name: str, basenames: list[str]):
        split_img_dir = root / split_name / images_postfix
        split_feat_dir = root / split_name / features_postfix
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_feat_dir.mkdir(parents=True, exist_ok=True)

        for b in basenames:
            src_img = img_dir / f"{b}.jpg"
            src_feat = feat_dir / f"{b}.pt"
            if not src_img.is_file() or not src_feat.is_file():
                print(f"[SKIP] Mancano file per {b}")
                continue
            safe_transfer(src_img, split_img_dir / src_img.name)
            safe_transfer(src_feat, split_feat_dir / src_feat.name)

    export_split("train", train_basenames)
    export_split("val", val_basenames)

    print(f"[INFO] Totale coppie: {n_total} | Train: {len(train_basenames)} | Val: {len(val_basenames)}")
    if move_instead_of_copy:
        print("[INFO] File originali spostati (non duplicati).")
    return {"train": train_basenames, "val": val_basenames}

@torch.no_grad()
def pca_features_to_rgb(
    feats: torch.Tensor,
    num_components: int = 3,
    outlier_rejection: bool = False,
    device: torch.device | str = "cpu",
    return_pil: bool = True,
    upsample_to_256: bool = False,
):
    """
    Converte feature map (B,C,H,W) o (H,W,C) in una visualizzazione RGB via PCA.
    
    Args:
        feats: Tensor [B,C,H,W] oppure [H,W,C] oppure [C,H,W].
        num_components: numero di componenti (ne useremo min(3, num_components) per RGB).
        outlier_rejection: se True applica robust scaling tipo median-based (come nel codice originale).
        device: 'cpu' o 'cuda'.
        return_pil: se True ritorna anche una Image PIL.
    Returns:
        rgb_tensor: torch.Tensor [H,W,3] in [0,1]
        pil_img (opzionale): PIL.Image
    """
    feats = feats.to(device)

    # Uniformiamo shape a (H,W,C_feat)
    if feats.dim() == 4:
        # assumiamo [B,C,H,W]
        if feats.shape[0] == 1:
            feats = feats[0]  # -> [C,H,W]
        else:
            raise ValueError("Supportata solo batch size = 1 per questa utility.")
    if feats.dim() == 3:
        # può essere [C,H,W] o [H,W,C]
        if feats.dim() == 3:
            # [C,H,W] tipico embedding: C > 10 e C != H
            if feats.shape[0] > 10 and feats.shape[0] != feats.shape[1]:
                feats = feats.permute(1, 2, 0)  # -> [H,W,C]
        elif feats.shape[-1] < 10:  # difficile ma per sicurezza
            pass  # già [H,W,C]
    else:
        raise ValueError("Dimensioni non supportate per feats.")

    H, W, C = feats.shape
    feats_flat = feats.reshape(-1, C).float()  # [H*W, C]

    # PCA (torch.pca_lowrank)
    q = max(num_components, 3)
    U, S, V = torch.pca_lowrank(feats_flat, q=q, center=True)

    proj = feats_flat @ V[:, :3]  # [H*W, 3]

    if outlier_rejection:
        # median absolute deviation like
        d = torch.abs(proj - torch.median(proj, dim=0).values)
        mdev = torch.median(d, dim=0).values.clamp(min=1e-6)
        s = d / mdev
        m = 2.0
        for k in range(3):
            valid = proj[s[:, k] < m, k]
            if valid.numel() > 0:
                proj[:, k] = torch.clamp(proj[:, k], valid.min(), valid.max())
        # normalizza
        proj -= proj.min(0, keepdim=True)[0]
        proj /= proj.max(0, keepdim=True)[0].clamp(min=1e-6)
    else:
        # min-max per canale
        proj -= proj.min(0, keepdim=True)[0]
        proj /= proj.max(0, keepdim=True)[0].clamp(min=1e-6)

    rgb = proj.reshape(H, W, 3).clamp(0, 1)

    if upsample_to_256:
        # Upsample to 256x256 if needed
        if rgb.shape[0] != 256 or rgb.shape[1] != 256:
            rgb = F.interpolate(
                rgb.permute(2, 0, 1).unsqueeze(0),  # [1, 3, H, W]
                size=(256, 256),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)  # [256, 256, 3]

    if return_pil:
        pil_img = Image.fromarray((rgb.cpu().numpy() * 255).astype("uint8"))
        return rgb, pil_img
    return rgb

def create_student_original_teacher_side_by_side(
    student_embeddings,
    teacher_embeddings,
    img_path,
    epoch,
    output_heatmaps,
    is_overfit_image=False,
    save_embeddings=False,
):
    """
    Visualizza teacher e student embeddings con colori coerenti.
    Se is_overfit_image=True → calcola la PCA dai teacher embeddings e la salva/carica localmente.
    Se False → calcola la PCA dinamicamente dai teacher embeddings (senza salvataggio/caricamento su disco).
    Salva anche gli embeddings se save_embeddings=True.
    """

    img_p = Path(img_path)
    img_name = img_p.stem
    local_basis_path = img_p.parent / f"{img_name}.pt"

    # --- Step 1: gestisci caricamento/salvataggio base PCA ---
    if is_overfit_image:
        if local_basis_path.exists():
            basis = torch.load(local_basis_path, map_location="cpu")
        else:
            print(f"[INFO] Computing PCA basis from teacher embeddings and saving to {local_basis_path}")
            feats = teacher_embeddings.clone().detach().to("cpu")
            if feats.dim() == 4:
                feats = feats[0]  # [C, H, W]
            feats = feats.permute(1, 2, 0).contiguous().reshape(-1, feats.shape[0])  # [H*W, C]
            U, S, V = torch.pca_lowrank(feats, q=3, center=True)
            basis = {"V": V[:, :3], "mean": feats.mean(0)}
            torch.save(basis, str(local_basis_path))
    else:
        feats = teacher_embeddings.clone().detach().to("cpu")
        if feats.dim() == 4:
            feats = feats[0]  # [C, H, W]
        feats = feats.permute(1, 2, 0).contiguous().reshape(-1, feats.shape[0])  # [H*W, C]
        U, S, V = torch.pca_lowrank(feats, q=3, center=True)
        basis = {"V": V[:, :3], "mean": feats.mean(0)}

    # --- Step 2: funzione helper per proiettare embeddings con la base caricata ---
    def project_with_basis(embeddings, basis):
        feats = embeddings.clone().detach().to("cpu")
        if feats.dim() == 4:
            feats = feats[0]
        feats = feats.permute(1, 2, 0).reshape(-1, feats.shape[0])  # [H*W, C]
        feats_centered = feats - basis["mean"]
        proj = feats_centered @ basis["V"]  # [H*W, 3]
        proj -= proj.min(0, keepdim=True)[0]
        proj /= proj.max(0, keepdim=True)[0].clamp(min=1e-6)
        H, W = embeddings.shape[-2:]
        rgb = proj.reshape(H, W, 3)
        pil_img = Image.fromarray((rgb.cpu().numpy() * 255).astype("uint8"))
        return pil_img

    # --- Step 3: proietta teacher e student sulla stessa base ---
    pil_img_teacher = project_with_basis(teacher_embeddings, basis)
    pil_img_student = project_with_basis(student_embeddings, basis)

    # --- Step 4: crea immagine combinata ---
    orig_img = Image.open(img_path).convert("RGB")
    target_size = orig_img.size
    pil_img_student = pil_img_student.resize(target_size, Image.BILINEAR)
    pil_img_teacher = pil_img_teacher.resize(target_size, Image.BILINEAR)
    w, h = target_size
    combined_img = Image.new("RGB", (w * 3, h))
    combined_img.paste(pil_img_student, (0, 0))
    combined_img.paste(orig_img, (w, 0))
    combined_img.paste(pil_img_teacher, (w * 2, 0))

    # Etichette
    draw = ImageDraw.Draw(combined_img)
    font = ImageFont.load_default(size=32)
    label_height = 40
    draw.rectangle([(0, 0), (w, label_height)], fill=(0, 0, 0, 128))
    draw.rectangle([(w, 0), (w * 2, label_height)], fill=(0, 0, 0, 128))
    draw.rectangle([(w * 2, 0), (w * 3, label_height)], fill=(0, 0, 0, 128))
    draw.text((10, 5), "STUDENT EMBEDDINGS", fill=(255, 255, 255), font=font)
    draw.text((w + 10, 5), "ORIGINAL IMAGE", fill=(255, 255, 255), font=font)
    draw.text((w * 2 + 10, 5), "TEACHER EMBEDDINGS", fill=(255, 255, 255), font=font)

    # --- Step 5: salva il risultato ---
    # include image base name together with epoch to make filenames unique
    combined_path = os.path.join(output_heatmaps, f"{epoch}_{img_name}.png")
    combined_img.save(combined_path)

    # --- Step 6: salva gli embeddings se richiesto ---
    if save_embeddings:
        student_dir = Path(output_heatmaps) / "student"
        teacher_dir = Path(output_heatmaps) / "teacher"
        student_dir.mkdir(parents=True, exist_ok=True)
        teacher_dir.mkdir(parents=True, exist_ok=True)
        torch.save(student_embeddings.detach().cpu(), student_dir / f"{epoch}.pt")
        torch.save(teacher_embeddings.detach().cpu(), teacher_dir / f"{epoch}.pt")

def mean_std_difference(student_embeddings, teacher_embeddings):
    """
    Computes and prints the mean and standard deviation of student and teacher embeddings for each batch,
    as well as the mean and standard deviation of their differences. Also calculates and prints the average
    difference between the means and standard deviations across all batches. Additionally computes and prints
    the average cosine similarity between student and teacher embeddings for each batch and overall.

    Args:
        student_embeddings (torch.Tensor): A batch of student embeddings of shape (B, ...), where B is the batch size.
        teacher_embeddings (torch.Tensor): A batch of teacher embeddings of shape (B, ...), where B is the batch size.

    Prints:
        For each batch:
            - Student mean and standard deviation
            - Teacher mean and standard deviation
            - Mean and standard deviation of the difference between student and teacher embeddings
            - Cosine similarity between student and teacher embeddings
        At the end:
            - Average difference between means across all batches
            - Average difference between standard deviations across all batches
            - Average cosine similarity across all batches
    """
    B = student_embeddings.shape[0]

    student_means = []
    student_stds = []
    teacher_means = []
    teacher_stds = []
    cosine_sims = []

    for i in range(B):
        s_mean = student_embeddings[i].mean().item()
        s_std = student_embeddings[i].std().item()
        t_mean = teacher_embeddings[i].mean().item()
        t_std = teacher_embeddings[i].std().item()
        student_means.append(s_mean)
        student_stds.append(s_std)
        teacher_means.append(t_mean)
        teacher_stds.append(t_std)

        # Cosine similarity calculation
        s_flat = student_embeddings[i].flatten().float()
        t_flat = teacher_embeddings[i].flatten().float()
        cos_sim = F.cosine_similarity(s_flat.unsqueeze(0), t_flat.unsqueeze(0)).item()
        cosine_sims.append(cos_sim)

    mean_diff = sum(student_means) / B - sum(teacher_means) / B
    std_diff = sum(student_stds) / B - sum(teacher_stds) / B
    avg_cosine_sim = sum(cosine_sims) / B

    return mean_diff, std_diff, avg_cosine_sim

def heatmap_sanity_check_single_channel(student_embeddings, teacher_embeddings, folder_name, output_dir):
    """
    Generates and saves side-by-side heatmap visualizations comparing the same randomly selected channel
    from the student and teacher embedding feature maps for all batch elements.

    The function upsamples both the student and teacher feature maps to the larger spatial resolution
    between the two, normalizes them to the [0, 1] range, and plots them as heatmaps for visual inspection.
    The resulting figures are saved to the specified output directory.

    Args:
        student_embeddings (torch.Tensor): Student feature maps of shape (B, C, H_s, W_s).
        teacher_embeddings (torch.Tensor): Teacher feature maps of shape (B, C, H_t, W_t).
        output_dir (str): Directory path where the heatmap images will be saved.

    Returns:
        None
    """
    B, C, H_s, W_s = student_embeddings.shape
    _, _, H_t, W_t = teacher_embeddings.shape

    channel_idx = random.randint(0, C - 1)

    target_H = max(H_s, H_t)
    target_W = max(W_s, W_t)

    for batch_idx in range(B):
        student_map = student_embeddings[batch_idx, channel_idx:channel_idx+1, :, :].unsqueeze(0)
        teacher_map = teacher_embeddings[batch_idx, channel_idx:channel_idx+1, :, :].unsqueeze(0)

        # Upsample to the larger resolution between student and teacher
        student_upsampled = F.interpolate(student_map, size=(target_H, target_W), mode='bilinear', align_corners=False).squeeze()
        teacher_upsampled = F.interpolate(teacher_map, size=(target_H, target_W), mode='bilinear', align_corners=False).squeeze()

        # Normalize to [0,1]
        student_norm = (student_upsampled - student_upsampled.min()) / (student_upsampled.max() - student_upsampled.min() + 1e-8)
        teacher_norm = (teacher_upsampled - teacher_upsampled.min()) / (teacher_upsampled.max() - teacher_upsampled.min() + 1e-8)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        sns.heatmap(student_norm.cpu().numpy(), ax=axs[0], cbar=True)
        axs[0].set_title(f"Student Embeddings - Batch {batch_idx}, Channel {channel_idx}")
        sns.heatmap(teacher_norm.cpu().numpy(), ax=axs[1], cbar=True)
        axs[1].set_title(f"Teacher Embeddings - Batch {batch_idx}, Channel {channel_idx}")

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{folder_name}_heatmap_batch{batch_idx}_channel{channel_idx}.png")
        plt.savefig(output_path)
        plt.close(fig)

        print(f"[DEBUG] Saved heatmap for batch {batch_idx}, channel {channel_idx} to {output_path}")

def heatmap_sanity_check_avg_all_channels(student_embeddings, teacher_embeddings, folder_name, output_dir):
    """
    Generates and saves side-by-side heatmap visualizations comparing the average feature maps
    across all channels from the student and teacher embeddings for all batch elements.

    The function upsamples both the student and teacher average feature maps to the larger spatial resolution
    between the two, normalizes them to the [0, 1] range, and plots them as heatmaps for visual inspection.
    The resulting figures are saved to the specified output directory.

    Args:
        student_embeddings (torch.Tensor): Student feature maps of shape (B, C, H_s, W_s).
        teacher_embeddings (torch.Tensor): Teacher feature maps of shape (B, C, H_t, W_t).
        output_dir (str): Directory path where the heatmap images will be saved.

    Returns:
        None
    """
    B, C, H_s, W_s = student_embeddings.shape
    _, _, H_t, W_t = teacher_embeddings.shape

    target_H = max(H_s, H_t)
    target_W = max(W_s, W_t)

    for batch_idx in range(B):
        student_map = student_embeddings[batch_idx].mean(dim=0, keepdim=True).unsqueeze(0)
        teacher_map = teacher_embeddings[batch_idx].mean(dim=0, keepdim=True).unsqueeze(0)

        # Upsample to the larger resolution between student and teacher
        student_upsampled = F.interpolate(student_map, size=(target_H, target_W), mode='bilinear', align_corners=False).squeeze()
        teacher_upsampled = F.interpolate(teacher_map, size=(target_H, target_W), mode='bilinear', align_corners=False).squeeze()

        # Normalize to [0,1]
        student_norm = (student_upsampled - student_upsampled.min()) / (student_upsampled.max() - student_upsampled.min() + 1e-8)
        teacher_norm = (teacher_upsampled - teacher_upsampled.min()) / (teacher_upsampled.max() - teacher_upsampled.min() + 1e-8)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        sns.heatmap(student_norm.cpu().numpy(), ax=axs[0], cbar=True)
        axs[0].set_title(f"Student Embeddings Avg All Channels - Batch {batch_idx}")
        sns.heatmap(teacher_norm.cpu().numpy(), ax=axs[1], cbar=True)
        axs[1].set_title(f"Teacher Embeddings Avg All Channels - Batch {batch_idx}")

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{folder_name}_heatmap_avg_all_channels_batch{batch_idx}.png")
        plt.savefig(output_path)
        plt.close(fig)

        print(f"[DEBUG] Saved heatmap for batch {batch_idx} (avg all channels) to {output_path}")

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

def verify_frozen_blocks(blocks, block_name="blocks", unfrozen_indices=None):
    """Verifica e stampa i blocchi ancora frozen."""
    frozen_indices = []
    for i in range(len(blocks)):
        is_frozen = any(not p.requires_grad for p in blocks[i].parameters())
        if is_frozen:
            frozen_indices.append(i)
    
    if frozen_indices:
        print(f"[VERIFY] ⚠️  {block_name} still FROZEN: {frozen_indices}")
    else:
        print(f"[VERIFY] ✅ All {len(blocks)} {block_name} are UNFROZEN")
    
    return frozen_indices

def print_trainable_summary(model, detailed=False, max_print=None):
    """
    Stampa un summary dettagliato dei parametri trainable raggruppati per modulo.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict con statistiche (trainable_count, frozen_count, trainable_groups)
    """
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
    
    stats = {
        "trainable_count": trainable_count,
        "frozen_count": frozen_count,
        "total_count": total_count,
        "trainable_groups": trainable_groups,
    }

    if detailed:
        log_param_status(model, max_print=max_print)
    
    return stats

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

def convert_mask_decoder_output_to_showable(
    masks_logits: torch.Tensor,
    iou_preds: torch.Tensor,
    mask_threshold: float = 0.0,
    return_format: str = "binary_mask",
) -> List[Dict[str, Any]]:
    """
    Converte l'output grezzo di MaskDecoder in formato visualizzabile con show_anns.
    
    Arguments:
      masks_logits (torch.Tensor): Output logit da MaskDecoder, shape (B, H, W) o (B, 1, H, W)
      iou_preds (torch.Tensor): Predizioni IoU da MaskDecoder, shape (B,) o (B, 1)
      mask_threshold (float): Threshold per binarizzare i logit
      return_format (str): "binary_mask" o "rle"
    
    Returns:
      list(dict): Lista di annotazioni compatibile con show_anns
    """
    from sam2_minimal.utils.amg import area_from_rle, mask_to_rle_pytorch, box_xyxy_to_xywh, batched_mask_to_box
    
    # Normalizza le dimensioni
    if masks_logits.dim() == 4:
        masks_logits = masks_logits.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
    
    if iou_preds.dim() == 2:
        iou_preds = iou_preds.squeeze(1)  # (B, 1) -> (B,)
    
    # Binarizza i logit - IMPORTANTE: deve essere bool, non float!
    masks_binary = masks_logits > mask_threshold  # <-- Rimosso .float()
    
    # Aggiungi dimensione batch se necessario per batched_mask_to_box
    if masks_binary.dim() == 2:
        masks_binary = masks_binary.unsqueeze(0)
    
    # Calcola i bounding box
    boxes = batched_mask_to_box(masks_binary)
    
    # Converti in RLE se necessario
    if return_format == "rle":
        rles = mask_to_rle_pytorch(masks_binary)
    
    # Costruisci le annotazioni
    anns = []
    for i, (mask, iou_pred, box) in enumerate(zip(masks_binary, iou_preds, boxes)):
        if return_format == "binary_mask":
            segmentation = mask.cpu().numpy()  # Già booleano
            area = int(segmentation.sum())
        else:  # rle
            segmentation = rles[i]
            area = area_from_rle(segmentation)
        
        ann = {
            "segmentation": segmentation,
            "area": area,
            "bbox": box_xyxy_to_xywh(box).tolist(),
            "predicted_iou": float(iou_pred.item()),
        }
        anns.append(ann)
    
    return anns

def convert_maskdecoder_to_showable(
    masks_logits: torch.Tensor,
    iou_preds: torch.Tensor,
    mask_threshold: float = 0.0,
    orig_size: tuple = None,
) -> List[Dict[str, Any]]:
    """
    Converte l'output grezzo di MaskDecoder in formato compatibile con show_anns.
    
    Args:
        masks_logits: Output logit da MaskDecoder, shape (B, C, H, W) o (B, H, W)
        iou_preds: Predizioni IoU da MaskDecoder, shape (B, C) o (B,)
        mask_threshold: Threshold per binarizzare i logit (default: 0.0)
        orig_size: Dimensione originale immagine (H, W) per resize. Se None, usa dimensione masks_logits
    
    Returns:
        Lista di dict compatibili con show_anns, con chiavi:
        - 'segmentation': maschera binaria numpy (H, W)
        - 'area': area in pixel
        - 'bbox': bounding box in formato XYWH
        - 'predicted_iou': predizione IoU del modello
    """
    
    # Normalizza dimensioni: porta tutto a (B, H, W)
    if masks_logits.dim() == 4:
        # (B, C, H, W) -> flatten primo e secondo asse
        B, C = masks_logits.shape[:2]
        masks_logits = masks_logits.reshape(B * C, *masks_logits.shape[2:])
        iou_preds = iou_preds.reshape(B * C)
    
    # Binarizza i logit (applica threshold)
    masks_binary = (masks_logits > mask_threshold)
    
    # Resize alle dimensioni originali se specificato
    if orig_size is not None:
        masks_binary = torch.nn.functional.interpolate(
            masks_binary.unsqueeze(1).float(),
            size=orig_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(1) > 0.5
    
    # Calcola bounding boxes
    boxes = batched_mask_to_box(masks_binary)
    
    # Converti in formato show_anns
    anns = []
    for i in range(len(masks_binary)):
        mask_np = masks_binary[i].cpu().numpy().astype(bool)
        
        ann = {
            'segmentation': mask_np,
            'area': int(mask_np.sum()),
            'bbox': box_xyxy_to_xywh(boxes[i]).tolist(),
            'predicted_iou': float(iou_preds[i].item()),
        }
        anns.append(ann)
    
    return anns

def setup_freeze_strategy(
    model: torch.nn.Module,
    num_info_sharing_blocks_unfreeze: int = 0,
    num_dino_layers_unfreeze: int = 0,
    student_components: list = None,
) -> dict:
    """
    Setup freeze/unfreeze strategy for distillation training.
    
    Args:
        model: MapAnything model
        num_info_sharing_blocks_unfreeze: Number of last info_sharing blocks to unfreeze
        num_dino_layers_unfreeze: Number of last DINOv2 encoder blocks to unfreeze
        student_components: List of module name prefixes to always unfreeze
                          (default: ["dpt_feature_head_2", "sam2_compat", "sam2_mask_decoder_student"])
    
    Returns:
        dict with keys:
            - "info_sharing_unfrozen_indices": list of unfrozen info_sharing block indices
            - "dino_unfrozen_indices": list of unfrozen DINOv2 block indices
    """
    if student_components is None:
        # student_components = ["dpt_feature_head_2", "sam2_compat", "sam2_mask_decoder_student"]
        student_components = ["dpt_feature_head_2", "sam2_compat"]
    
    # 1. Freeze tutto
    print("[INFO] Freezing all parameters...")
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. Unfreeze componenti student
    print(f"[INFO] Unfreezing student components: {student_components}")
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in student_components):
            param.requires_grad = True
    
    result = {
        "info_sharing_unfrozen_indices": [],
        "dino_unfrozen_indices": [],
    }
    
    # 3. Unfreeze info_sharing blocks
    if num_info_sharing_blocks_unfreeze > 0 and hasattr(model, "info_sharing"):
        info_sharing = model.info_sharing
        blocks = getattr(info_sharing, "self_attention_blocks", None)
        
        if blocks and len(blocks) > 0:
            start_idx = max(0, len(blocks) - num_info_sharing_blocks_unfreeze)
            unfrozen_indices = []
            unfrozen_count = 0
            
            for i in range(start_idx, len(blocks)):
                for param in blocks[i].parameters():
                    param.requires_grad = True
                    unfrozen_count += param.numel()
                unfrozen_indices.append(i)
            
            # Unfreeze wrapper params if all blocks unfrozen
            if num_info_sharing_blocks_unfreeze == 24:
                for name, p in model.named_parameters():
                    if name.startswith(("info_sharing.proj_embed", "info_sharing.norm")):
                        p.requires_grad = True
            
            result["info_sharing_unfrozen_indices"] = unfrozen_indices
            print(f"[INFO] Unfroze last {num_info_sharing_blocks_unfreeze} info_sharing blocks (indices {unfrozen_indices})")
            print(f"[INFO] Unfroze {unfrozen_count:,} parameters in info_sharing")
            
            from nico.utils import verify_frozen_blocks
            verify_frozen_blocks(blocks, "Multi-View Transformer blocks", unfrozen_indices)
    
    # 4. Unfreeze DINOv2 blocks
    if num_dino_layers_unfreeze > 0 and hasattr(model, "encoder"):
        dino_encoder = model.encoder
        dino_model = getattr(dino_encoder, "model", dino_encoder)
        blocks = getattr(dino_model, "blocks", None)
        
        if blocks and len(blocks) > 0:
            start_idx = max(0, len(blocks) - num_dino_layers_unfreeze)
            unfrozen_indices = []
            unfrozen_count = 0
            
            for i in range(start_idx, len(blocks)):
                for param in blocks[i].parameters():
                    param.requires_grad = True
                    unfrozen_count += param.numel()
                unfrozen_indices.append(i)
            
            # Unfreeze wrapper params if all blocks unfrozen
            if num_dino_layers_unfreeze == 24:
                for name, p in model.named_parameters():
                    if name.startswith((
                        "encoder.model.patch_embed.proj",
                        "encoder.model.pos_embed",
                        "encoder.model.cls_token",
                        "encoder.model.norm",
                    )):
                        p.requires_grad = True
            
            result["dino_unfrozen_indices"] = unfrozen_indices
            print(f"[INFO] Unfroze last {num_dino_layers_unfreeze} DINOv2 encoder blocks (indices {unfrozen_indices})")
            print(f"[INFO] Unfroze {unfrozen_count:,} parameters in DINOv2 encoder")
            
            from nico.utils import verify_frozen_blocks
            verify_frozen_blocks(blocks, "DINOv2 encoder blocks", unfrozen_indices)
        else:
            print("[WARN] DINOv2 encoder has no 'blocks'. Skipping unfreezing.")
    
    return result

class EncoderDistillationLoss(torch.nn.Module):
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


# ==================== Checkpoint Management ====================
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
        and not getattr(getattr(args, "train_params", None) or {}, "override_scheduler", False)
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
) -> Tuple[int, float]:
    """
    Load checkpoint containing only ENCODER trainable components.
    
    Loads:
        - dpt_feature_head_2 (student encoder)
        - sam2_compat (student encoder compatibility layer)
        - unfrozen info_sharing blocks (multi-view transformer)
        - unfrozen DINOv2 encoder blocks
    
    Args:
        model_without_ddp: Model without DDP wrapper
        checkpoint_path: Path to encoder checkpoint
        device: Device to load checkpoint on
        args: Training arguments (optional, not required for loading)
    
    Returns:
        (start_epoch, best_val_loss) tuple from checkpoint
    """
    print(f"[LOAD] Loading encoder checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"[DEBUG] Checkpoint content: {ckpt.keys()}")
    
    # Load encoder head
    if "dpt_feature_head_2" in ckpt:
        model_without_ddp.dpt_feature_head_2.load_state_dict(ckpt["dpt_feature_head_2"])
        print("[INFO] Loaded dpt_feature_head_2")
    else:
        print("[WARN] dpt_feature_head_2 not found in encoder checkpoint!")
    
    # Load sam2_compat if present
    if "sam2_compat" in ckpt and hasattr(model_without_ddp, "sam2_compat"):
        model_without_ddp.sam2_compat.load_state_dict(ckpt["sam2_compat"])
        print("[INFO] Loaded sam2_compat")
    elif hasattr(model_without_ddp, "sam2_compat"):
        print("[INFO] sam2_compat not in encoder checkpoint; using random initialization")
    
    # Load info_sharing blocks (NO dipendenza da args)
    if "info_sharing_blocks" in ckpt and hasattr(model_without_ddp, "info_sharing"):
        info = model_without_ddp.info_sharing
        blocks = getattr(info, "self_attention_blocks", None)
        if blocks:
            for idx, state_dict in ckpt["info_sharing_blocks"].items():
                if idx < len(blocks):
                    try:
                        blocks[idx].load_state_dict(state_dict)
                    except Exception as e:
                        print(f"[WARN] Failed loading info_sharing block {idx}: {e}")
            print(f"[INFO] Restored {len(ckpt['info_sharing_blocks'])} info_sharing blocks")
        
        if "info_sharing_wrappers" in ckpt:
            for name, data in ckpt["info_sharing_wrappers"].items():
                try:
                    param = dict(info.named_parameters())[name]
                    param.data.copy_(data)
                except Exception as e:
                    print(f"[WARN] Failed loading wrapper param {name}: {e}")
            print(f"[INFO] Restored {len(ckpt['info_sharing_wrappers'])} info_sharing wrapper params")
    
    # Load DINOv2 blocks (NO dipendenza da args)
    if "dino_encoder_blocks" in ckpt and hasattr(model_without_ddp, "encoder"):
        dino_encoder = model_without_ddp.encoder
        dino_model = dino_encoder.model if hasattr(dino_encoder, "model") else dino_encoder
        blocks = getattr(dino_model, "blocks", None)
        if blocks:
            for idx, state_dict in ckpt["dino_encoder_blocks"].items():
                if idx < len(blocks):
                    try:
                        blocks[idx].load_state_dict(state_dict)
                    except Exception as e:
                        print(f"[WARN] Failed loading DINOv2 block {idx}: {e}")
            print(f"[INFO] Restored {len(ckpt['dino_encoder_blocks'])} DINOv2 blocks")
        
        if "dino_encoder_wrappers" in ckpt:
            for name, data in ckpt["dino_encoder_wrappers"].items():
                try:
                    param = dict(dino_model.named_parameters())[name]
                    param.data.copy_(data)
                except Exception as e:
                    print(f"[WARN] Failed loading wrapper param {name}: {e}")
            print(f"[INFO] Restored {len(ckpt['dino_encoder_wrappers'])} DINOv2 wrapper params")
    
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    
    return start_epoch, best_val_loss

def load_decoder_checkpoint(
    model_without_ddp,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[int, float]:
    """
    Load checkpoint containing only DECODER trainable components.
    
    Loads:
        - sam2_mask_decoder_student (student decoder)
    
    Args:
        model_without_ddp: Model without DDP wrapper
        checkpoint_path: Path to decoder checkpoint
        device: Device to load checkpoint on
    
    Returns:
        (start_epoch, best_val_loss) tuple from checkpoint
    """
    print(f"[LOAD] Loading decoder checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load decoder
    if "sam2_mask_decoder_student" in ckpt and hasattr(model_without_ddp, "sam2_mask_decoder_student"):
        model_without_ddp.sam2_mask_decoder_student.load_state_dict(ckpt["sam2_mask_decoder_student"])
        print("[INFO] Loaded sam2_mask_decoder_student")
    elif hasattr(model_without_ddp, "sam2_mask_decoder_student"):
        print("[WARN] sam2_mask_decoder_student not found in decoder checkpoint; using random initialization")
    else:
        print("[WARN] sam2_mask_decoder_student not present in model!")
    
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    
    return start_epoch, best_val_loss

def save_trainer_checkpoint(
    optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    output_dir: str,
    tag: str = "last",
    args=None,
    wandb_available: bool = False,
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
        args: Training arguments (for full reproducibility)
    """
    state = {
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    
    # Save full args for reproducibility
    if args is not None:
        state["args"] = args
    
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    
    if wandb_available and wandb.run is not None:
        state["wandb_run_id"] = wandb.run.id
    
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_trainer_{tag}.pth"
    torch.save(state, ckpt_path)
    print(f"[SAVE] Trainer checkpoint saved: {ckpt_path}")

def save_encoder_checkpoint(
    model_without_ddp,
    epoch: int,
    best_val_loss: float,
    output_dir: str,
    tag: str = "last",
    args=None,
    wandb_available: bool = False,
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
        epoch: Current epoch
        best_val_loss: Best validation loss so far
        output_dir: Directory to save checkpoint
        tag: Tag for checkpoint filename (e.g., "best", "last", "epoch10")
        args: Training arguments (optional, for reproducibility)
        wandb_available: Whether wandb is available
    """
    state = {
        "dpt_feature_head_2": model_without_ddp.dpt_feature_head_2.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    
    # Save full args for reproducibility
    if args is not None:
        state["args"] = args

    # Save sam2_compat if present
    if hasattr(model_without_ddp, "sam2_compat"):
        state["sam2_compat"] = model_without_ddp.sam2_compat.state_dict()
        print("[INFO] Added sam2_compat to encoder checkpoint")

    # === Determina blocchi unfrozen da stato effettivo del modello ===
    if hasattr(model_without_ddp, "info_sharing"):
        info = model_without_ddp.info_sharing
        blocks = getattr(info, "self_attention_blocks", None)
        if blocks:
            # Trova blocchi con requires_grad=True
            unfrozen_indices = [
                i for i, block in enumerate(blocks)
                if any(p.requires_grad for p in block.parameters())
            ]
            
            if unfrozen_indices:
                state["info_sharing_blocks"] = {
                    i: blocks[i].state_dict() for i in unfrozen_indices
                }
                print(f"[INFO] Saved {len(unfrozen_indices)} unfrozen info_sharing blocks: {unfrozen_indices}")
            
            # Salva wrapper params se unfrozen
            wrapper_state = {
                name: param.data.clone()
                for name, param in info.named_parameters()
                if param.requires_grad and name.startswith(("proj_embed", "norm"))
            }
            if wrapper_state:
                state["info_sharing_wrappers"] = wrapper_state
                print(f"[INFO] Saved {len(wrapper_state)} unfrozen info_sharing wrapper params")

    # === Stessa logica per DINOv2 ===
    if hasattr(model_without_ddp, "encoder"):
        dino_encoder = model_without_ddp.encoder
        dino_model = dino_encoder.model if hasattr(dino_encoder, "model") else dino_encoder
        blocks = getattr(dino_model, "blocks", None)
        if blocks:
            unfrozen_indices = [
                i for i, block in enumerate(blocks)
                if any(p.requires_grad for p in block.parameters())
            ]
            
            if unfrozen_indices:
                state["dino_encoder_blocks"] = {
                    i: blocks[i].state_dict() for i in unfrozen_indices
                }
                print(f"[INFO] Saved {len(unfrozen_indices)} unfrozen DINOv2 blocks: {unfrozen_indices}")
            
            wrapper_state = {
                name: param.data.clone()
                for name, param in dino_model.named_parameters()
                if param.requires_grad and name.startswith(("patch_embed.proj", "pos_embed", "cls_token", "norm"))
            }
            if wrapper_state:
                state["dino_encoder_wrappers"] = wrapper_state
                print(f"[INFO] Saved {len(wrapper_state)} unfrozen DINOv2 wrapper params")
    
    # Save wandb run_id if available
    if wandb_available and wandb.run is not None:
        state["wandb_run_id"] = wandb.run.id
    
    # Crea la sottocartella checkpoints
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = ckpt_dir / f"checkpoint_encoder_{tag}.pth"
    torch.save(state, ckpt_path)
    print(f"[SAVE] Encoder checkpoint saved: {ckpt_path}")

def save_decoder_checkpoint(
    model_without_ddp,
    epoch: int,
    best_val_loss: float,
    output_dir: str,
    tag: str = "last",
    args=None,
    wandb_available: bool = False,
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
        epoch: Current epoch
        best_val_loss: Best validation loss so far
        output_dir: Directory to save checkpoint
        tag: Tag for checkpoint filename (e.g., "best", "last", "epoch10")
        args: Training arguments (optional, for reproducibility)
        wandb_available: Whether wandb is available
    """
    state = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    
    # Save full args for reproducibility
    if args is not None:
        state["args"] = args

    # Save student mask decoder if present
    if hasattr(model_without_ddp, "sam2_mask_decoder_student"):
        state["sam2_mask_decoder_student"] = model_without_ddp.sam2_mask_decoder_student.state_dict()
        print("[INFO] Added sam2_mask_decoder_student to decoder checkpoint")
    else:
        print("[WARN] sam2_mask_decoder_student not found in model!")
    
    # Save wandb run_id if available
    if wandb_available and wandb.run is not None:
        state["wandb_run_id"] = wandb.run.id
    
    # Crea la sottocartella checkpoints
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = ckpt_dir / f"checkpoint_decoder_{tag}.pth"
    torch.save(state, ckpt_path)
    print(f"[SAVE] Decoder checkpoint saved: {ckpt_path}")

def save_model(
    model_without_ddp,
    epoch: int,
    best_val_loss: float,
    output_dir: str,
    tag: str,
    optimizer=None,
    scheduler=None,
    args=None,
    wandb_available: bool = False,
    save_encoder: bool = True,
    save_decoder: bool = True,
    save_trainer: bool = True,
):
    """
    Unified checkpoint saving function that saves encoder, decoder, and trainer state.
    
    This is a convenience wrapper that calls save_encoder_checkpoint(), save_decoder_checkpoint(),
    and save_trainer_checkpoint() based on the provided flags.
    
    Args:
        model_without_ddp: Model without DDP wrapper
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        best_val_loss: Best validation loss so far
        output_dir: Directory to save checkpoints
        tag: Tag for checkpoint filenames (e.g., "best", "last", "epoch10")
        args: Training arguments (for reproducibility)
        wandb_available: Whether wandb is available
        save_encoder: Whether to save encoder checkpoint
        save_decoder: Whether to save decoder checkpoint
        save_trainer: Whether to save trainer checkpoint
    """
    if save_encoder:
        save_encoder_checkpoint(
            model_without_ddp=model_without_ddp,
            epoch=epoch,
            best_val_loss=best_val_loss,
            output_dir=output_dir,
            tag=tag,
            args=args,
            wandb_available=wandb_available,
        )
    
    if save_decoder:
        save_decoder_checkpoint(
            model_without_ddp=model_without_ddp,
            epoch=epoch,
            best_val_loss=best_val_loss,
            output_dir=output_dir,
            tag=tag,
            args=args,
            wandb_available=wandb_available,
        )
    
    if save_trainer:
        save_trainer_checkpoint(
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            output_dir=output_dir,
            tag=tag,
            args=args,
            wandb_available=wandb_available,
        )

def load_model(
    model_without_ddp,
    device: torch.device,
    optimizer=None,
    scheduler=None,
    args=None,
    encoder_checkpoint_path: str = None,
    decoder_checkpoint_path: str = None,
    trainer_checkpoint_path: str = None,
) -> Tuple[int, float]:
    """
    Unified checkpoint loading function that loads encoder, decoder, and trainer state.
    
    This is a convenience wrapper that calls load_encoder_checkpoint(), load_decoder_checkpoint(),
    and load_trainer_checkpoint() as needed, consolidating the start_epoch and best_val_loss.
    
    Args:
        model_without_ddp: Model without DDP wrapper
        device: Device to load checkpoints on
        optimizer: Optimizer (for loading optimizer state from trainer checkpoint)
        scheduler: Scheduler (for loading scheduler state from trainer checkpoint)
        args: Training arguments with override flags
        encoder_checkpoint_path: Path to encoder checkpoint (optional)
        decoder_checkpoint_path: Path to decoder checkpoint (optional)
        trainer_checkpoint_path: Path to trainer checkpoint (optional)
    
    Returns:
        (start_epoch, best_val_loss): Consolidated values from all loaded checkpoints
            - start_epoch: max epoch from all checkpoints + 1
            - best_val_loss: min loss from all checkpoints
    """
    start_epoch = 0
    best_val_loss = float("inf")
    
    loaded_checkpoints = []
    
    # Load encoder checkpoint
    if encoder_checkpoint_path and os.path.exists(encoder_checkpoint_path):
        enc_epoch, enc_loss = load_encoder_checkpoint(
            model_without_ddp=model_without_ddp,
            checkpoint_path=encoder_checkpoint_path,
            device=device,
        )
        start_epoch = max(start_epoch, enc_epoch)
        best_val_loss = min(best_val_loss, enc_loss)
        loaded_checkpoints.append(f"encoder (epoch {enc_epoch}, loss {enc_loss:.6f})")
    
    # Load decoder checkpoint
    if decoder_checkpoint_path and os.path.exists(decoder_checkpoint_path):
        dec_epoch, dec_loss = load_decoder_checkpoint(
            model_without_ddp=model_without_ddp,
            checkpoint_path=decoder_checkpoint_path,
            device=device,
        )
        start_epoch = max(start_epoch, dec_epoch)
        best_val_loss = min(best_val_loss, dec_loss)
        loaded_checkpoints.append(f"decoder (epoch {dec_epoch}, loss {dec_loss:.6f})")
    
    # Load trainer checkpoint
    if trainer_checkpoint_path and os.path.exists(trainer_checkpoint_path):
        trainer_epoch, trainer_loss = load_trainer_checkpoint(
            checkpoint_path=trainer_checkpoint_path,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
        )
        start_epoch = max(start_epoch, trainer_epoch)
        best_val_loss = min(best_val_loss, trainer_loss)
        loaded_checkpoints.append(f"trainer (epoch {trainer_epoch}, loss {trainer_loss:.6f})")
    
    # Summary
    if loaded_checkpoints:
        print(f"\n[RESUME] Loaded {len(loaded_checkpoints)} checkpoint(s): {', '.join(loaded_checkpoints)}")
        print(f"[RESUME] Consolidated: start_epoch={start_epoch}, best_val_loss={best_val_loss:.6f}\n")
    else:
        print("[INFO] No checkpoints loaded, starting from scratch")
    
    return start_epoch, best_val_loss

def pca_visualization(student_features, teacher_features):
    try:
        import matplotlib
        matplotlib.use('Agg') # Fondamentale per SSH
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        import os
        import numpy as np

        output_path = "/scratch2/nico/distillation/test_visivo"
        os.makedirs(output_path, exist_ok=True)
        
        # Prendi il primo elemento del batch se necessario
        # pred_sem e gt_sem sono attesi come (B, C, H, W) -> prendiamo (C, H, W)
        s_tensor = student_features[0] if student_features.ndim == 4 else student_features
        t_tensor = teacher_features[0] if teacher_features.ndim == 4 else teacher_features

        # Converti in NumPy (H, W, C)
        s_feat = s_tensor.detach().cpu().permute(1, 2, 0).numpy()
        t_feat = t_tensor.detach().cpu().permute(1, 2, 0).numpy()

        H, W, C = s_feat.shape
        
        # Appiattisci per PCA
        s_flat = s_feat.reshape(-1, C)
        t_flat = t_feat.reshape(-1, C)

        # PCA a 3 componenti (RGB)
        pca = PCA(n_components=3)
        
        # FIT SOLO SUL TEACHER! 
        # Questo garantisce che "rosso" significhi la stessa cosa per entrambi
        pca.fit(t_flat)

        s_pca = pca.transform(s_flat).reshape(H, W, 3)
        t_pca = pca.transform(t_flat).reshape(H, W, 3)

        # Normalizzazione Min-Max indipendente (0-1) per visualizzazione
        def normalize_vis(x):
            mi, ma = x.min(), x.max()
            return (x - mi) / (ma - mi + 1e-8)

        s_rgb = normalize_vis(s_pca)
        t_rgb = normalize_vis(t_pca)

        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        ax[0].imshow(s_rgb)
        ax[0].set_title(f"Teacher Prediction")
        ax[0].axis('off')
        
        ax[1].imshow(t_rgb)
        ax[1].set_title("Student Target")
        ax[1].axis('off')

        # Salva con nome univoco (usa un timestamp o un contatore globale se disponibile)
        import time
        timestamp = int(time.time() * 1000)
        save_path = os.path.join(output_path, f"vis_debug_{timestamp}.png")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"[DEBUG VIS] Saved PCA comparison to {save_path}")

    except Exception as e:
        print(f"[DEBUG VIS ERROR] {e}")