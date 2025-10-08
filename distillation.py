# Optional config for better memory efficiency
"""
Simple single-GPU distillation script:
Per ogni cartella:
  - carica tutte le immagini come un batch multi-view
  - carica le teacher features (una per immagine) già salvate su disco
  - esegue un forward del modello (in modalità training) per ottenere gli embeddings della head aggiuntiva
  - calcola una loss MSE tra embeddings student e teacher (dopo eventuale pooling / normalizzazione)

Assunzioni:
  - La head aggiuntiva (instance_head) è stata integrata in MapAnything e salva le sue uscite in
    res[i]["inst_embeddings_8x"] durante il forward (NON infer, perché infer è in no_grad).
  - Le teacher features sono salvate con nome: <nome_immagine>_features.pt nella stessa cartella (o modificare pattern).
  - Ogni file teacher contiene un tensore shape (C, Ht, Wt) o (1, C, Ht, Wt).
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import wandb

from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from nico.my_head import InstanceSegmentationHead
from nico.utils import mean_std_difference, heatmap_sanity_check_single_channel, heatmap_sanity_check_avg_all_channels, create_student_original_teacher_side_by_side
import random
from tqdm import tqdm

# ==================== CONFIGURAZIONE MANUALE ====================
# Modifica qui i parametri invece di passare argomenti da CLI
INPUT_DIR = "/scratch2/nico/distillation/training_samples"           # Directory che contiene sottocartelle di immagini
OUTPUT_DIR = "/scratch2/nico/distillation/output"         # Directory per log / checkpoint
COCO2017_ROOT = "/scratch2/nico/distillation/coco2017"  # root che contiene 'train' e 'val'
IMAGES_DIRNAME = "val2017"              # sottocartella immagini dentro ogni split
FEATURES_DIRNAME = "teacher_features"   # sottocartella features dentro ogni split
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TRAIN_IMAGES_DIR = os.path.join(COCO2017_ROOT, TRAIN_SPLIT, IMAGES_DIRNAME)
VAL_IMAGES_DIR = os.path.join(COCO2017_ROOT, VAL_SPLIT, IMAGES_DIRNAME)
TRAIN_FEATURES_DIR = os.path.join(COCO2017_ROOT, TRAIN_SPLIT, FEATURES_DIRNAME)
VAL_FEATURES_DIR = os.path.join(COCO2017_ROOT, VAL_SPLIT, FEATURES_DIRNAME)
EPOCHS = 50                                 # Numero di epoche - insensatamente alto ma tanto c'è early stopping
LR = 1e-4                                   # Learning rate
WEIGHT_DECAY = 0.0                          # Weight decay AdamW
EMB_POOL_SIZE = 64                          # (Non usato direttamente ora, placeholder se estendi pooling custom)
SEED = 0                                    # Seed random
AMP = True                                  # Abilita autocast mixed precision
NORM = False                                # Normalizza embeddings prima della loss
SINGLE_IMAGE = True                         # Carica e processa una immagine per volta (batch size 1)
BATCH_SIZE_IMAGES = 1                       # Numero di immagini per batch (per sfruttare meglio la GPU)
DEBUG_MAX_TRAIN_IMAGES = 1000               # <= usa solo immagini campionate a caso in train (None o 0 per disabilitare)
DEBUG_MAX_VAL_IMAGES = 50                   # opzionale: limita anche la val (None o 0 per disabilitare)
# ===============================================================
# Riprendi da checkpoint (se non None)
# LOAD_CHECKPOINT = "checkpoint_epoch24.pth"  # es: "checkpoint_final.pth" oppure None
LOAD_CHECKPOINT = None
# ===============================================================
# Early stopping e ReduceLROnPlateau (impostare a True/False per abilitare/disabilitare)
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5  # epoche senza miglioramento prima di fermare
USE_LR_ON_PLATEAU = True
LR_ON_PLATEAU_PATIENCE = 3   # epoche senza miglioramento prima di ridurre LR
LR_ON_PLATEAU_FACTOR = 0.5   # fattore di riduzione LR
MIN_LR = 1e-7                # learning rate minimo consentito
# =================================================================

def save_checkpoint(model, optimizer, epoch, loss, output_dir, tag="last"):
    # Salva solo la instance_head e l'optimizer
    state = {
        "instance_head": model.instance_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    ckpt_path = Path(output_dir) / f"checkpoint_{tag}.pth"
    torch.save(state, ckpt_path)
    print(f"[INFO] Checkpoint salvato (solo head): {ckpt_path}")

def is_image_file(name: str) -> bool:
    name_low = name.lower()
    return name_low.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(SEED) # numpy seed
    torch.manual_seed(SEED) # torch seed

    if OUTPUT_DIR:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True) # crea cartella output se non esiste

    wandb.init(
        project="Run 2 - mapanything-distillation",
        notes="test con meno immagini per epoca per plottare piu spesso, head più profonda, AMP=True",
        config={
            "learning_rate": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE_IMAGES,
            "norm": NORM,
            "amp": AMP,
            "use_early_stopping": USE_EARLY_STOPPING,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "use_lr_on_plateau": USE_LR_ON_PLATEAU,
            "lr_on_plateau_patience": LR_ON_PLATEAU_PATIENCE,
            "lr_on_plateau_factor": LR_ON_PLATEAU_FACTOR,
            "min_lr": MIN_LR,
            "single_image": SINGLE_IMAGE,
            "debug_max_train_images": DEBUG_MAX_TRAIN_IMAGES,
            "debug_max_val_images": DEBUG_MAX_VAL_IMAGES,
            "loss_mix": "0.5_mse_0.5_cosine",
            "train_mean_diff": None,
            "train_std_diff": None,
            "train_cos_sim": None
        }
    )

    print(f"output_dir: {OUTPUT_DIR}")

    # Modello + freeze
    model = MapAnything.from_pretrained("facebook/map-anything").to(device)
    model.instance_head = InstanceSegmentationHead(in_dim=256).to(device)
    if not hasattr(model, "instance_head"):
        raise AttributeError("Il modello non ha 'instance_head'.")
    for name, p in model.named_parameters():
        if not name.startswith("instance_head"):
            p.requires_grad = False

    # Optimizer (solo head)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    # print(optimizer)

    # Scheduler ReduceLROnPlateau opzionale
    if USE_LR_ON_PLATEAU:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=LR_ON_PLATEAU_FACTOR, patience=LR_ON_PLATEAU_PATIENCE, min_lr=MIN_LR
        )

    # Caricamento checkpoint se richiesto
    start_epoch = 0
    best_loss = None # per early stopping
    epochs_no_improve = 0 # contatore early stopping
    if LOAD_CHECKPOINT is not None:
        ckpt_path = Path(OUTPUT_DIR) / LOAD_CHECKPOINT
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} non trovato!")
        checkpoint = torch.load(ckpt_path, map_location=device) # carica su device
        model.instance_head.load_state_dict(checkpoint["instance_head"]) # carica solo head
        optimizer.load_state_dict(checkpoint["optimizer"]) # carica stato optimizer
        start_epoch = checkpoint.get("epoch", 0) # riprendi da epoch successiva
        best_loss = checkpoint.get("loss", None)
        print(f"[INFO] Checkpoint {ckpt_path.name} caricato. Riprendo da epoch {start_epoch+1}.")

    print(f"Start training for {EPOCHS} epochs from epoch {start_epoch+1}.")
    start_time = time.time()

    # Prepara file di log con header corretto in base alla modalità
    if EPOCHS > 0:
        log_path = Path(OUTPUT_DIR) / "loss_log.csv"
        if not log_path.exists():
            with open(log_path, "w") as f:
                writer = csv.writer(f)
                if SINGLE_IMAGE:
                    # D = Down, U = Up - segnala direzione desiderata, aggiunti lr e epoch_time_sec
                    writer.writerow([
                        "epoch",
                        "train_loss - D",
                        "val_loss - D",
                        "train_mean_diff - D",
                        "train_std_diff - D",
                        "train_cos_sim - U (1.0)",
                        "mean_diff - D",
                        "std_diff - D",
                        "avg_cosine_sim - U (1.0)",
                        "lr",
                        "epoch_time_sec"
                    ])
                else:
                    writer.writerow(["epoch", "mean_loss - D", "mean_diff - D", "std_diff - D", "avg_cosine_sim - U (1.0)", "lr", "epoch_time_sec"])

    if SINGLE_IMAGE:
        print("[DEBUG] SINGLE_IMAGE=True: batching immagini singole come dimensione batch.")

        last_train_loss = None
        last_val_loss = None

        # ---- Carica liste train / val ----
        if not os.path.isdir(TRAIN_IMAGES_DIR) or not os.path.isdir(TRAIN_FEATURES_DIR):
            print(f"[ERR] Directory train mancanti: {TRAIN_IMAGES_DIR} / {TRAIN_FEATURES_DIR}")
            return
        if not os.path.isdir(VAL_IMAGES_DIR) or not os.path.isdir(VAL_FEATURES_DIR):
            print(f"[ERR] Directory val mancanti: {VAL_IMAGES_DIR} / {VAL_FEATURES_DIR}")
            return

        train_image_paths = [os.path.join(TRAIN_IMAGES_DIR, f) for f in os.listdir(TRAIN_IMAGES_DIR) if is_image_file(f)]
        val_image_paths = [os.path.join(VAL_IMAGES_DIR, f) for f in os.listdir(VAL_IMAGES_DIR) if is_image_file(f)]
        train_image_paths.sort()
        val_image_paths.sort()
        # Limita il numero di immagini per il debug: campionamento casuale ad ogni epoca
        train_image_paths_full = train_image_paths.copy()
        if DEBUG_MAX_VAL_IMAGES and len(val_image_paths) > DEBUG_MAX_VAL_IMAGES:
            val_image_paths = val_image_paths[:DEBUG_MAX_VAL_IMAGES]
            print(f"[DEBUG] Limito val a {len(val_image_paths)} immagini.")

        print(f"[SPLIT] Train images: {len(train_image_paths)} | Val images: {len(val_image_paths)}")
        if len(train_image_paths) == 0 or len(val_image_paths) == 0:
            print("[ERR] Uno degli split è vuoto.")
            return

        autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        for epoch in range(start_epoch, EPOCHS):
            epoch_t0 = time.time()
            print(f"Epoch {epoch+1}/{EPOCHS}")
            # ---- TRAIN ----
            # Campiona DEBUG_MAX_TRAIN_IMAGES immagini casuali dal training set
            if DEBUG_MAX_TRAIN_IMAGES and len(train_image_paths_full) > DEBUG_MAX_TRAIN_IMAGES:
                train_image_paths = random.sample(train_image_paths_full, DEBUG_MAX_TRAIN_IMAGES)
                print(f"[DEBUG] Epoca {epoch+1}: campiono {len(train_image_paths)} immagini train casuali.")
            else:
                train_image_paths = train_image_paths_full.copy()
            model.train(True)
            train_loss_acc = 0.0
            train_samples = 0
            train_mean_diff_acc = 0.0
            train_std_diff_acc = 0.0
            train_cos_sim_acc = 0.0
            train_mse_loss_acc = 0.0
            train_cos_loss_acc = 0.0
            # shuffle train
            random.shuffle(train_image_paths)
            for start_idx in tqdm(range(0, len(train_image_paths), BATCH_SIZE_IMAGES), desc=f"Train Ep {epoch+1}"):
                batch_paths = train_image_paths[start_idx:start_idx + BATCH_SIZE_IMAGES]
                views_list = load_images(batch_paths)
                if len(views_list) == 0:
                    continue
                imgs = torch.cat([v["img"] for v in views_list], dim=0).to(device, non_blocking=True)
                batched_view = [{"img": imgs, "data_norm_type": views_list[0]["data_norm_type"]}]
                # teacher batch
                teacher_tensors = []
                skip = False
                for p in batch_paths:
                    base = os.path.splitext(os.path.basename(p))[0]
                    t_path = Path(TRAIN_FEATURES_DIR) / f"{base}.pt"
                    if not t_path.is_file():
                        skip = True; break
                    t = torch.load(str(t_path), map_location="cpu")
                    if t.dim() == 3: t = t.unsqueeze(0)
                    if t.dim() != 4 or t.shape[0] != 1:
                        skip = True; break
                    teacher_tensors.append(t)
                if skip or len(teacher_tensors) == 0:
                    continue
                teacher_batch = torch.cat(teacher_tensors, dim=0).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type='cuda', enabled=AMP, dtype=autocast_dtype):
                    outputs = model.forward(batched_view)
                    o = outputs[0]
                    if "inst_embeddings_8x" not in o:
                        raise KeyError("inst_embeddings_8x mancante nell'output (train).")
                    student_batch = o["inst_embeddings_8x"]
                    if teacher_batch.shape[1:] != student_batch.shape[1:]:
                        raise ValueError(f"Shape mismatch teacher {teacher_batch.shape} vs student {student_batch.shape}")
                    if NORM:
                        student_norm = F.normalize(student_batch, dim=1) # normalization is needed to align embeddings representations onto a common space (hypersphere)
                        teacher_norm = F.normalize(teacher_batch, dim=1)
                    else:
                        student_norm = student_batch
                        teacher_norm = teacher_batch

                    cos_loss = 1 - F.cosine_similarity(student_norm, teacher_norm, dim=1).mean() # cosine loss for "teaching the same semantic language"
                    mse_loss = F.mse_loss(student_norm, teacher_norm) # mse loss for the same spatial structure/shape
                    loss = 0.5 * mse_loss + 0.5 * cos_loss
                loss.backward()
                optimizer.step()
                train_loss_acc += float(loss.detach().cpu()) * student_batch.shape[0]
                train_samples += student_batch.shape[0]
                # Accumula mse_loss e cos_loss pesati per numero immagini
                train_mse_loss_acc += float(mse_loss.detach().cpu()) * student_batch.shape[0]
                train_cos_loss_acc += float(cos_loss.detach().cpu()) * student_batch.shape[0]
                # Metriche train (accumulate pesate per numero immagini)
                md, sd, cs = mean_std_difference(student_batch, teacher_batch)
                train_mean_diff_acc += float(md) * student_batch.shape[0]
                train_std_diff_acc  += float(sd) * student_batch.shape[0]
                train_cos_sim_acc   += float(cs) * student_batch.shape[0]
            train_loss_mean = train_loss_acc / train_samples if train_samples > 0 else 0.0
            train_mse_loss_mean = train_mse_loss_acc / train_samples if train_samples > 0 else 0.0
            train_cos_loss_mean = train_cos_loss_acc / train_samples if train_samples > 0 else 0.0
            train_mean_diff = train_mean_diff_acc / train_samples if train_samples > 0 else 0.0
            train_std_diff  = train_std_diff_acc  / train_samples if train_samples > 0 else 0.0
            train_cos_sim   = train_cos_sim_acc   / train_samples if train_samples > 0 else 0.0

            # ---- VALIDATION ----
            model.eval()
            val_loss_acc = 0.0
            val_samples = 0
            val_mean_diff_acc = 0.0
            val_std_diff_acc = 0.0
            val_cos_sim_acc = 0.0
            val_mse_loss_acc = 0.0
            val_cos_loss_acc = 0.0
            with torch.no_grad():
                for start_idx in tqdm(range(0, len(val_image_paths), BATCH_SIZE_IMAGES), desc=f"Val Ep {epoch+1}"):
                    batch_paths = val_image_paths[start_idx:start_idx + BATCH_SIZE_IMAGES]
                    views_list = load_images(batch_paths)
                    if len(views_list) == 0:
                        continue
                    imgs = torch.cat([v["img"] for v in views_list], dim=0).to(device, non_blocking=True)
                    batched_view = [{"img": imgs, "data_norm_type": views_list[0]["data_norm_type"]}]
                    teacher_tensors = []
                    skip = False
                    for p in batch_paths:
                        base = os.path.splitext(os.path.basename(p))[0]
                        t_path = Path(VAL_FEATURES_DIR) / f"{base}.pt"
                        if not t_path.is_file(): skip = True; break
                        t = torch.load(str(t_path), map_location="cpu")
                        if t.dim() == 3: t = t.unsqueeze(0)
                        if t.dim() != 4 or t.shape[0] != 1: skip = True; break
                        teacher_tensors.append(t)
                    if skip or len(teacher_tensors) == 0: continue
                    teacher_batch = torch.cat(teacher_tensors, dim=0).to(device)
                    with torch.amp.autocast(device_type='cuda', enabled=AMP, dtype=autocast_dtype):
                        outputs = model.forward(batched_view)
                        o = outputs[0]
                        if "inst_embeddings_8x" not in o:
                            raise KeyError("inst_embeddings_8x mancante nell'output (val).")
                        student_batch = o["inst_embeddings_8x"]
                        if teacher_batch.shape[1:] != student_batch.shape[1:]:
                            raise ValueError(f"Shape mismatch (val) teacher {teacher_batch.shape} vs student {student_batch.shape}")
                        if NORM:
                            student_norm = F.normalize(student_batch, dim=1)
                            teacher_norm = F.normalize(teacher_batch, dim=1)
                        else:
                            student_norm = student_batch
                            teacher_norm = teacher_batch

                        cos_loss = 1 - F.cosine_similarity(student_norm, teacher_norm, dim=1).mean()
                        mse_loss = F.mse_loss(student_norm, teacher_norm)
                        vloss = 0.5 * mse_loss + 0.5 * cos_loss
                    val_loss_acc += float(vloss.detach().cpu()) * student_batch.shape[0]
                    val_samples += student_batch.shape[0]
                    val_mse_loss_acc += float(mse_loss.detach().cpu()) * student_batch.shape[0]
                    val_cos_loss_acc += float(cos_loss.detach().cpu()) * student_batch.shape[0]
                    md, sd, cs = mean_std_difference(student_batch, teacher_batch)
                    val_mean_diff_acc += float(md) * student_batch.shape[0]
                    val_std_diff_acc  += float(sd) * student_batch.shape[0]
                    val_cos_sim_acc   += float(cs) * student_batch.shape[0]
            val_loss_mean = val_loss_acc / val_samples if val_samples > 0 else 0.0
            val_mean_diff = val_mean_diff_acc / val_samples if val_samples > 0 else 0.0
            val_std_diff  = val_std_diff_acc  / val_samples if val_samples > 0 else 0.0
            val_cos_sim   = val_cos_sim_acc   / val_samples if val_samples > 0 else 0.0
            val_mse_loss_mean = val_mse_loss_acc / val_samples if val_samples > 0 else 0.0
            val_cos_loss_mean = val_cos_loss_acc / val_samples if val_samples > 0 else 0.0
            last_train_loss = train_loss_mean
            last_val_loss = val_loss_mean

            epoch_time = time.time() - epoch_t0
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1} done - "
                f"train_loss={train_loss_mean:.4f} val_loss={val_loss_mean:.4f} | "
                f"val_mean_diff={val_mean_diff:.4f} val_std_diff={val_std_diff:.4f} "
                f"val_cos_sim={val_cos_sim:.4f} lr={current_lr:.2e} time={epoch_time:.1f}s"
            )

            # Log CSV
            if EPOCHS > 0:
                with open(log_path, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch+1,
                        train_loss_mean,
                        val_loss_mean,
                        train_mean_diff,
                        train_std_diff,
                        train_cos_sim,
                        val_mean_diff,
                        val_std_diff,
                        val_cos_sim,
                        current_lr,
                        f"{epoch_time:.2f}"
                    ])

            # wandb logging
            wandb.log({
                "epoch": epoch+1,
                "train_loss": train_loss_mean if 'train_loss_mean' in locals() else epoch_loss_mean,
                "train_mse_loss": train_mse_loss_mean if 'train_mse_loss_mean' in locals() else None,
                "train_cos_loss": train_cos_loss_mean if 'train_cos_loss_mean' in locals() else None,
                "train_mean_diff": train_mean_diff,
                "train_std_diff": train_std_diff,
                "train_cos_sim": train_cos_sim,
                "val_loss": val_loss_mean if 'val_loss_mean' in locals() else None,
                "val_mse_loss": val_mse_loss_mean if 'val_mse_loss_mean' in locals() else None,
                "val_cos_loss": val_cos_loss_mean if 'val_cos_loss_mean' in locals() else None,
                "mean_diff": val_mean_diff if 'val_mean_diff' in locals() else mean_diff,
                "std_diff": val_std_diff if 'val_std_diff' in locals() else std_diff,
                "cosine_similarity": val_cos_sim if 'val_cos_sim' in locals() else avg_cosine_sim,
                "lr": current_lr,
                "epoch_time_sec": epoch_time
            })

            # Scheduler & Early Stopping sulla val_loss
            target_loss = val_loss_mean
            if USE_LR_ON_PLATEAU:
                scheduler.step(target_loss)
            improved = best_loss is None or target_loss < best_loss - 1e-6
            if improved:
                best_loss = target_loss; epochs_no_improve = 0
                save_checkpoint(model, optimizer, epoch+1, target_loss, OUTPUT_DIR, tag="best")
            else:
                epochs_no_improve += 1
                print(f"[INFO] Nessun miglioramento val ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE}).")
            if USE_EARLY_STOPPING and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"[EARLY STOP] Stoppo a epoch {epoch+1}.")
                break

            save_checkpoint(model, optimizer, epoch+1, target_loss, OUTPUT_DIR, tag=f"epoch{epoch+1}")

    else:
        print("[DEBUG] SINGLE_IMAGE=False: Carico e processo tutte le immagini per ogni cartella.")

        folders = [f for f in sorted(Path(INPUT_DIR).iterdir()) if f.is_dir()]
        if len(folders) == 0:
            print(f"Nessuna cartella trovata in {INPUT_DIR}")
            return

        autocast_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        for epoch in range(start_epoch, EPOCHS):
            epoch_loss_acc = 0.0 # somma delle loss per epoch
            samples_acc = 0 # numero di immagini totali per epoch
            epoch_t0 = time.time()
            print(f"Epoch {epoch+1}/{EPOCHS}")
            model.train(True) # modalità training (abilita dropout, batchnorm, ecc.)

            for folder in tqdm(folders, desc=f"Epoch {epoch+1}"):
                # Path cartella immagini
                img_dir = str(folder)

                # Preprocessing immagini
                views = load_images(img_dir)
                if len(views) == 0:
                    print(f"[WARN] Cartella vuota: {folder}")
                    continue

                # Carica un unico file teacher_embeddings.pt con shape (B, C, H, W)
                teacher_path = Path(folder) / "teacher_embeddings.pt"
                if not teacher_path.exists():
                    print(f"[SKIP] teacher_embeddings.pt non trovato in {folder.name}")
                    continue
                teacher_batch = torch.load(teacher_path, map_location="cpu") # carica su CPU per risparmiare memoria GPU (sposto dopo su GPU solo il batch che mi serve)
                if teacher_batch.dim() != 4:
                    print(f"[WARN] Shape teacher inattesa: {teacher_batch.shape}")
                    continue
                if teacher_batch.shape[0] != len(views):
                    print(f"[WARN] Mismatch batch: teacher ({teacher_batch.shape[0]}) vs immagini ({len(views)}) in {folder.name}")
                    continue

                # Sposta immagini su device (in-place)
                for v in views:
                    if isinstance(v, dict) and "img" in v:
                        v["img"] = v["img"].to(device, non_blocking=True) # non_blocking=True per trasferimento asincrono

                teacher_batch = teacher_batch.to(device) # sposta batch teacher su GPU

                optimizer.zero_grad(set_to_none=True) # azzera i gradienti prima di calcolare quelli nuovi (set_to_none=True per efficienza memoria)

                with torch.amp.autocast(device_type='cuda', enabled=AMP, dtype=autocast_dtype):
                    outputs = model.forward(views) # produce embeddings dell'encoder di MapAnything (in training mode, quindi calcola i gradienti)
                    student_list = []
                    for o in outputs:
                        if "inst_embeddings_8x" not in o:
                            raise KeyError("Output della head 'inst_embeddings_8x' non trovato. Verifica integrazione.")
                        emb = o["inst_embeddings_8x"] # filtra gli emebddings della head aggiuntiva
                        student_list.append(emb)
                    student_batch = torch.cat(student_list, dim=0) # concatena in un unico batch gli embeddings (B, C, H, W)

                    if teacher_batch.shape[1] != student_batch.shape[1]: # non dovrebbe mai succedere
                        raise ValueError(
                            f"Mismatch canali teacher({teacher_batch.shape[1]}) vs student({student_batch.shape[1]}). Aggiungi un proiettore.")

                    if NORM:
                        student_norm = F.normalize(student_batch, dim=1)
                        teacher_norm = F.normalize(teacher_batch, dim=1)
                    else:
                        student_norm = student_batch
                        teacher_norm = teacher_batch

                    cos_loss = 1 - F.cosine_similarity(student_norm, teacher_norm, dim=1).mean()
                    mse_loss = F.mse_loss(student_norm, teacher_norm)
                    loss = 0.5 * mse_loss + 0.5 * cos_loss


                loss.backward() # calcola i gradienti
                optimizer.step() # aggiorna i pesi della head in base ai gradienti calcolati

                batch_loss = float(loss.detach().cpu()) # sposta la loss su CPU e converte in float
                epoch_loss_acc += batch_loss * len(views) # somma la loss pesata per il numero di immagini
                samples_acc += len(views) # aggiorna il contatore immagini
                print(f"[Folder {folder.name}] N={len(views)} loss={batch_loss:.4f}")

                # Analisi statistiche (opzionale, commenta se non serve)
                mean_diff, std_diff, avg_cosine_sim = mean_std_difference(student_batch, teacher_batch)
                # print(f"  Mean diff: {mean_diff:.6f}, Std diff: {std_diff:.6f}, Avg Cosine Sim: {avg_cosine_sim:.6f}")

            epoch_loss_mean = epoch_loss_acc / samples_acc if samples_acc > 0 else 0.0 # loss media epoch
            epoch_time = time.time() - epoch_t0
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1} done - mean_loss={epoch_loss_mean:.4f} mean_diff={mean_diff:.4f} std_diff={std_diff:.4f} avg_cosine_sim={avg_cosine_sim:.4f} lr={current_lr:.2e} time={epoch_time/60:.2f} min")

            # Logga la loss su file CSV
            with open(log_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch+1,
                    epoch_loss_mean,
                    mean_diff,
                    std_diff,
                    avg_cosine_sim,
                    current_lr,
                    f"{epoch_time:.2f}"
                ])

            # wandb logging
            wandb.log({
                "epoch": epoch+1,
                "train_loss": train_loss_mean if 'train_loss_mean' in locals() else epoch_loss_mean,
                "train_mse_loss": mse_loss.item() if 'mse_loss' in locals() else None,
                "train_cos_loss": cos_loss.item() if 'cos_loss' in locals() else None,
                "val_loss": val_loss_mean if 'val_loss_mean' in locals() else None,
                "mean_diff": val_mean_diff if 'val_mean_diff' in locals() else mean_diff,
                "std_diff": val_std_diff if 'val_std_diff' in locals() else std_diff,
                "cosine_similarity": val_cos_sim if 'val_cos_sim' in locals() else avg_cosine_sim,
                "lr": current_lr,
                "epoch_time_sec": epoch_time
            })

            # Scheduler LR on plateau
            if USE_LR_ON_PLATEAU:
                scheduler.step(epoch_loss_mean) # step dello scheduler passando la loss media

            # Early stopping
            improved = best_loss is None or epoch_loss_mean < best_loss - 1e-6
            if improved:
                best_loss = epoch_loss_mean
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"[INFO] Nessun miglioramento da {epochs_no_improve} epoche.")
            if USE_EARLY_STOPPING and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"[EARLY STOPPING] Stoppo il training dopo {epoch+1} epoche: nessun miglioramento su loss per {EARLY_STOPPING_PATIENCE} epoche.")
                break

            # Salva checkpoint ogni epoca
            save_checkpoint(model, optimizer, epoch+1, epoch_loss_mean, OUTPUT_DIR, tag=f"epoch{epoch+1}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} min")

    # Salva modello finale
    if EPOCHS > 0:
        if SINGLE_IMAGE:
            final_loss = last_val_loss if (last_val_loss is not None) else (last_train_loss if last_train_loss is not None else 0.0)
        else:
            final_loss = epoch_loss_mean if 'epoch_loss_mean' in locals() else (best_loss if best_loss is not None else 0.0)
        final_epoch = (epoch + 1) if 'epoch' in locals() else start_epoch
        save_checkpoint(model, optimizer, final_epoch, final_loss, OUTPUT_DIR, tag="final")

    # Analisi finale con heatmap
    if SINGLE_IMAGE:
        # Usa preferibilmente lo split di validazione per le heatmap finali; fallback al train se mancano.
        source_image_paths = None
        if 'val_image_paths' in locals() and len(val_image_paths) > 0:
            source_image_paths = val_image_paths
            print("[HEATMAP] Uso immagini di validation per l'analisi finale.")
        elif 'train_image_paths' in locals() and len(train_image_paths) > 0:
            source_image_paths = train_image_paths
            print("[HEATMAP] Val vuoto: uso immagini di train per l'analisi finale.")
        else:
            print("[HEATMAP][WARN] Nessuna lista immagini disponibile per generare heatmap.")
            source_image_paths = []

        num_images = min(10, len(source_image_paths))
        if num_images == 0:
            print("[HEATMAP][WARN] Nessuna immagine da processare.")
        else:
            selected_paths = random.sample(source_image_paths, num_images)
            print(f"Analisi heatmap per {num_images} immagini scelte casualmente dallo split selezionato.")

            for img_path in selected_paths:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                print(f"Analisi heatmap per immagine: {img_name}")
                # Caricamento singola immagine mantenendo compatibilità con load_images
                view = load_images([str(img_path)])
                if len(view) == 0:
                    print(f"[WARN] Nessuna immagine caricata da {img_path}")
                    continue
                if isinstance(view, dict):
                    view = [view]  # normalizza a lista
                for v in view:
                    if "img" in v:
                        v["img"] = v["img"].to(device, non_blocking=True)

                model.eval()
                model.infer(
                    view,
                    memory_efficient_inference=False,
                    use_amp=True,
                    amp_dtype="bf16",
                    apply_mask=True,
                    mask_edges=True,
                    apply_confidence_mask=False,
                    confidence_percentile=0,
                )
                student_embeddings = getattr(model, "_last_inst_embeddings", None)
                if student_embeddings is None:
                    print("[HEATMAP][WARN] _last_inst_embeddings non presente dopo infer().")
                    continue

                # Path teacher: prima prova nello split di validazione, poi fallback al train
                candidate_teacher_paths = [
                    os.path.join(VAL_FEATURES_DIR, f"{img_name}.pt"),
                    os.path.join(TRAIN_FEATURES_DIR, f"{img_name}.pt"),
                ]
                teacher_path = None
                for ctp in candidate_teacher_paths:
                    if os.path.isfile(ctp):
                        teacher_path = ctp
                        break
                if teacher_path is None:
                    print(f"[HEATMAP][WARN] Teacher feature non trovata per {img_name}.")
                    continue
                teacher_embeddings = torch.load(teacher_path, map_location="cpu")

                output_heatmaps = os.path.join(OUTPUT_DIR, "heatmaps")
                os.makedirs(output_heatmaps, exist_ok=True)
                print(f"[HEATMAP] Salvo heatmap in {output_heatmaps}")
                # heatmap_sanity_check_single_channel(student_embeddings, teacher_embeddings, img_name, output_heatmaps)
                # heatmap_sanity_check_avg_all_channels(student_embeddings, teacher_embeddings, img_name, output_heatmaps)
                create_student_original_teacher_side_by_side(student_embeddings, teacher_embeddings, img_path, img_name, output_heatmaps)
    else:
        for folder in folders:
            print(f"Analisi heatmap per cartella: {folder.name}")
            images = os.path.join(INPUT_DIR, folder)
            views = load_images(images)

            model.eval() # modalità eval (disabilita dropout, batchnorm, ecc.)
            model.infer(
                views,
                memory_efficient_inference=False,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
                confidence_percentile=0,
            )
            student_embeddings = getattr(model, "_last_inst_embeddings", None)

            teacher_path = Path(folder) / "teacher_embeddings.pt"
            teacher_embeddings = torch.load(teacher_path, map_location="cpu")

            output_heatmaps = os.path.join(OUTPUT_DIR, "heatmaps")
            print(f"[DEBUG] Saving heatmaps to {output_heatmaps}")
            os.makedirs(output_heatmaps, exist_ok=True)
            # heatmap_sanity_check_single_channel(student_embeddings, teacher_embeddings, folder.name, output_heatmaps)
            # heatmap_sanity_check_avg_all_channels(student_embeddings, teacher_embeddings, folder.name, output_heatmaps)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrotto manualmente da tastiera.")
    except Exception as e:
        print(f"[ERROR] Eccezione imprevista: {e}")
        raise
    finally:
        # chiusura sicura di wandb
        print("[CLEANUP] Chiusura sessione wandb e salvataggio stato finale.")
        wandb.finish()