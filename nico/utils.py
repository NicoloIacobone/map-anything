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

# def create_student_original_teacher_side_by_side(
#     student_embeddings,
#     teacher_embeddings,
#     img_path,
#     img_name,
#     output_heatmaps,
# ):
#     _, pil_img_student = pca_features_to_rgb(student_embeddings, num_components=3, outlier_rejection=False)
#     _, pil_img_teacher = pca_features_to_rgb(teacher_embeddings, num_components=3, outlier_rejection=False)
#     # Resize both images to the same size if needed
#     # Load original image
#     orig_img = Image.open(img_path).convert("RGB")

#     # Upsample student and teacher images to match original image size if needed
#     target_size = orig_img.size
#     if pil_img_student.size != target_size:
#         pil_img_student = pil_img_student.resize(target_size, Image.BILINEAR)
#     if pil_img_teacher.size != target_size:
#         pil_img_teacher = pil_img_teacher.resize(target_size, Image.BILINEAR)

#     # Create a new image with triple width to place student, original, and teacher side by side
#     w, h = target_size
#     combined_img = Image.new("RGB", (w * 3, h))
#     combined_img.paste(pil_img_student, (0, 0))
#     combined_img.paste(orig_img, (w, 0))
#     combined_img.paste(pil_img_teacher, (w * 2, 0))

#     # Add labels to each image

#     draw = ImageDraw.Draw(combined_img)
#     font = ImageFont.load_default(size=32)

#     label_height = 40
#     # Draw rectangles for label backgrounds
#     draw.rectangle([(0, 0), (w, label_height)], fill=(0, 0, 0, 128))
#     draw.rectangle([(w, 0), (w * 2, label_height)], fill=(0, 0, 0, 128))
#     draw.rectangle([(w * 2, 0), (w * 3, label_height)], fill=(0, 0, 0, 128))

#     draw.text((10, 5), "STUDENT EMBEDDINGS", fill=(255, 255, 255), font=font)   
#     draw.text((w + 10, 5), "ORIGINAL IMAGE", fill=(255, 255, 255), font=font)
#     draw.text((w * 2 + 10, 5), "TEACHER EMBEDDINGS", fill=(255, 255, 255), font=font)

#     # Save the combined image
#     # combined_path = os.path.join(output_heatmaps, f"{img_name}_student_original_teacher_side_by_side.png")
#     combined_path = os.path.join(output_heatmaps, f"{img_name}.png")
#     combined_img.save(combined_path)
#     # print(f"[INFO] Saved side-by-side image: {combined_path}")

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
            feats = feats.permute(1, 2, 0).reshape(-1, feats.shape[0])  # [H*W, C]
            U, S, V = torch.pca_lowrank(feats, q=3, center=True)
            basis = {"V": V[:, :3], "mean": feats.mean(0)}
            torch.save(basis, str(local_basis_path))
    else:
        feats = teacher_embeddings.clone().detach().to("cpu")
        if feats.dim() == 4:
            feats = feats[0]  # [C, H, W]
        feats = feats.permute(1, 2, 0).reshape(-1, feats.shape[0])  # [H*W, C]
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

    # debug - print shapes of teacher and student pil images
    print(f"[DEBUG] Teacher PIL image size: {pil_img_teacher.size}, Student PIL image size: {pil_img_student.size}")

    # --- Step 4: crea immagine combinata ---
    orig_img = Image.open(img_path).convert("RGB")
    target_size = orig_img.size
    pil_img_student = pil_img_student.resize(target_size, Image.BILINEAR)
    pil_img_teacher = pil_img_teacher.resize(target_size, Image.BILINEAR)

    # debug - print shapes of teacher and student resized images
    print(f"[DEBUG] Resized Teacher PIL image size: {pil_img_teacher.size}, Resized Student PIL image size: {pil_img_student.size}")

    raise Exception("Debug stop")

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
    combined_path = os.path.join(output_heatmaps, f"{epoch}.png")
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