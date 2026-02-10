import datetime
# import json
# import math
import os
# import pickle
# import sys
import time
# from collections import defaultdict
from pathlib import Path
from typing import Sized, Optional
from PIL import Image

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

# Visualizzazione PCA
import matplotlib
matplotlib.use('Agg') # Fondamentale per SSH / No-GUI environment
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import mapanything.utils.train_tools as train_tools
from mapanything.datasets import get_test_data_loader, get_train_data_loader
from mapanything.models import init_model
from mapanything.train.losses import *  # noqa
from mapanything.utils.inference import loss_of_one_batch_multi_view
from mapanything.utils.train_tools import NativeScalerWithGradNormCount as NativeScaler
from nico.utils import (
    print_trainable_summary,
    setup_freeze_strategy,
    load_model,
    save_model,
)
from sam2_minimal.sam2_builder import (
    build_sam_mask_decoder,
    # load_sam2_feature_extractor,
    load_sam2_teacher_prompt_and_decoder,
)
from mapanything.distill.help_me.dataset_dataloader import (
    TeacherFeatureExtractor,
    # DistillationDataset,
    # collate_fn_distillation,
    # build_distillation_dataloader,
)

# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True

def distillation(args):
    # Initialize distributed training if required
    train_tools.init_distributed_mode(args.distributed)
    global_rank = train_tools.get_rank()

    # Init output directory and device
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        if args.train_params.run_name and global_rank == 0:
            Path(os.path.join(args.output_dir, args.train_params.run_name)).mkdir(parents=True, exist_ok=True)

    # Print all arguments if required
    # print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(", ", ",\n"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix the seed
    seed = args.train_params.seed + train_tools.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.train_params.disable_cudnn_benchmark

    # ========== BUILD DATALOADERS ==========
    print("Building train dataset {:s}".format(args.dataset.train_dataset))
    data_loader_train = build_dataset(
        dataset=args.dataset.train_dataset,
        num_workers=args.dataset.num_workers,
        test=False,
        max_num_of_imgs_per_gpu=args.train_params.max_num_of_imgs_per_gpu,
    )
    print("Building test dataset {:s}".format(args.dataset.test_dataset))
    test_batch_size = 2 * (
        args.train_params.max_num_of_imgs_per_gpu // args.dataset.num_views
    )  # Since we don't have any backward overhead
    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset=dataset,
            num_workers=args.dataset.num_workers,
            test=True,
            batch_size=test_batch_size,
        )
        for dataset in args.dataset.test_dataset.split("+")
        if "(" in dataset
    }

    # ========== LOAD MODEL + STUDENT ENCODER ==========
    if global_rank == 0:
        model = init_model(
            args.model.model_str,
            args.model.model_config,
            torch_hub_force_reload=args.model.torch_hub_force_reload,
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # Make sure the model is initialized before proceeding
    if global_rank != 0:
        model = init_model(
            args.model.model_str, args.model.model_config, torch_hub_force_reload=False
        )
    model.to(device)  # Move model to device
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

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

    teacher_extractor = TeacherFeatureExtractor(
        checkpoint_path=args.sam2_path,
        device=str(device),
        # augment_cfg=augment_cfg,
        augment_cfg = None
    )
    teacher_extractor.to(device)

    # ========== INITIALIZE TEACHER DECODER ==========
    print(f"[INFO] Loading teacher PromptEncoder and MaskDecoder...")
    sam_prompt_encoder_teacher, sam_mask_decoder_teacher = load_sam2_teacher_prompt_and_decoder(
        checkpoint_path=args.sam2_path,
        device=str(device),
        image_size=1024,
        backbone_stride=16,
        embed_dim=256,
    )
    print(f"[INFO] Teacher decoder components loaded and frozen")

    # ========== FREEZE STRATEGY ==========
    freeze_info = setup_freeze_strategy(
        model=model,
        num_info_sharing_blocks_unfreeze=args.train_params.num_info_sharing_blocks_unfreeze,
        num_dino_layers_unfreeze=args.train_params.num_dino_layers_unfreeze,
    )

    # ========== VERIFY TRAINABLE PARAMETERS ==========
    if args.train_params.print_trainable:
        print_trainable_summary(model, detailed=True)

    # ========== CRITERION ==========
    # Criterion
    print(f">> Creating train criterion = {args.loss.train_criterion}")
    train_criterion = eval(args.loss.train_criterion).to(device)
    print(
        f">> Creating test criterion = {args.loss.test_criterion or args.loss.train_criterion}"
    )
    test_criterion = eval(args.loss.test_criterion or args.loss.train_criterion).to(device)    

    # ========== WRAPPING IN DDP ==========
    if args.distributed.distributed:
        print("[INFO] Wrapping model in DistributedDataParallel (DDP)...")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.distributed.gpu],
            find_unused_parameters=False, # perché tutti i moduli trainable sono sempre usati nella forward
            static_graph=False, # il grafo cambia perché usiamo diversi numeri di views in diverse scene, in single view va bene False
        )
        model_without_ddp = model.module

    # ========== OPTIMIZER con LR differenziati e LOSS SCALER ==========
    # Following timm: set wd as 0 for bias and norm layers
    param_groups, param_groups_name_to_idx_map, param_groups_idx_to_name_map = (
        train_tools.get_parameter_groups(
            model_without_ddp,
            args.train_params.base_lr,
            args.train_params.weight_decay,
            submodule_configs=args.train_params.submodule_configs,
            warn_not_in_submodule=True,
        )
    )
    optimizer = torch.optim.AdamW(
        param_groups, betas=(0.9, 0.95)
    )
    loss_scaler = NativeScaler()

    # Stampa riepilogo dei gruppi di parametri
    if args.train_params.print_optim_params:
        print("[INFO] Optimizer param groups:")
        submodule_totals = {}
        for submodule_name in param_groups_name_to_idx_map:
            indices = param_groups_name_to_idx_map[submodule_name]
            total_params = 0
            lr = None
            for idx in indices:
                lr = param_groups[idx]['lr']
                num_params = sum(p.numel() for p in param_groups[idx]['params'])
                total_params += num_params
            submodule_totals[submodule_name] = (total_params, lr)
            print(f"  {submodule_name}: {total_params:,} params @ LR {lr:.6e}")

    # ========== LEARNING RATE SCHEDULER ==========
    scheduler = None
    if args.train_params.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.train_params.lr_scheduler_t_max,
            eta_min=args.train_params.lr_min,
        )
        print(f"[INFO] Using CosineAnnealingLR with T_max={args.train_params.lr_scheduler_t_max}, eta_min={args.train_params.lr_min}")
    elif args.train_params.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.train_params.lr_decay_epochs,
            gamma=0.1,
        )
        print(f"[INFO] Using StepLR with step_size={args.train_params.lr_decay_epochs}, gamma=0.1")
    elif args.train_params.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=args.train_params.plateau_patience,
            threshold=1e-3,
            threshold_mode='abs',
            cooldown=1,
            min_lr=args.train_params.lr_min,
        )
        print(f"[INFO] Using ReduceLROnPlateau (factor=0.5, patience={args.train_params.plateau_patience}, threshold=1e-3 abs, cooldown=1)")
    else:
        print(f"[INFO] Learning rate scheduler disabled. Base LR will remain constant at {args.train_params.base_lr}")

    # ========= RESUME FROM CHECKPOINT (IF SPECIFIED) ==========
    start_epoch = args.train_params.start_epoch
    best_val_loss = float("inf")
    best_so_far = best_val_loss # per tenere traccia del best durante il training

    if (args.train_params.resume_encoder_ckpt or 
        args.train_params.resume_decoder_ckpt or 
        args.train_params.resume_trainer_ckpt):
        
        start_epoch, best_val_loss = load_model(
            model_without_ddp=model_without_ddp,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
        )

    # Override start_epoch if explicitly set in args
    if args.train_params.start_epoch > 0:
        start_epoch = args.train_params.start_epoch
        print(f"[INFO] Overriding start_epoch from args: {start_epoch}")
 
    # Handle LR override after loading checkpoints
    if getattr(args.train_params, "override_lr", False):
        # Ricalcola i LR usando i valori direttamente da submodule_configs
        for submodule_name in param_groups_name_to_idx_map:
            indices = param_groups_name_to_idx_map[submodule_name]
            # I valori di LR sono già impostati correttamente da get_parameter_groups()
            # Non serve ricalcolare - il YAML ha già i valori finali
            pass  # Il LR è già corretto dal YAML
        
        lr_info = ", ".join([
            f"{name}={optimizer.param_groups[param_groups_name_to_idx_map[name][0]]['lr']:.6e}"
            for name in param_groups_name_to_idx_map
        ])
        print(f"[INFO] Optimizer LR configured from submodule_configs (YAML): {lr_info}")
    
    # Scheduler advance logic if we resumed from a checkpoint
    if args.train_params.resume_encoder_ckpt or args.train_params.resume_decoder_ckpt or args.train_params.resume_trainer_ckpt:
        if getattr(args.train_params, "override_scheduler", False) and scheduler is not None:
            resumed_epoch = start_epoch
            
            if args.train_params.lr_scheduler == "step":
                # StepLR: Chiama step() per ogni epoca già completata
                for _ in range(resumed_epoch):
                    scheduler.step()
                # Stampa tutti i LR dopo l'advance
                lr_info = ", ".join([
                    f"{name}={optimizer.param_groups[param_groups_name_to_idx_map[name][0]]['lr']:.6e}"
                    for name in param_groups_name_to_idx_map
                ])
                print(f"[INFO] Advanced StepLR scheduler by {resumed_epoch} steps (LRs: {lr_info})")
            
            elif args.train_params.lr_scheduler == "cosine":
                # CosineAnnealingLR: Salta direttamente all'epoca corrente
                scheduler.last_epoch = resumed_epoch - 1  # last_epoch è 0-indexed
                scheduler.step()  # Aggiorna LR basato su last_epoch
                # Stampa tutti i LR dopo l'advance
                lr_info = ", ".join([
                    f"{name}={optimizer.param_groups[param_groups_name_to_idx_map[name][0]]['lr']:.6e}"
                    for name in param_groups_name_to_idx_map
                ])
                print(f"[INFO] Set CosineAnnealingLR to epoch {resumed_epoch} (LRs: {lr_info})")
        
        if start_epoch > 0:
            print(f"[RESUME] Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    # |=======================================================================================================================================|
    # |============================================================ TRAINING LOOP ============================================================|
    # |=======================================================================================================================================|
    print(f"[INFO] Start distillation training for {args.train_params.epochs} epochs from epoch {start_epoch}")
    start_time = time.time()

    for epoch in range(args.train_params.start_epoch, args.train_params.epochs + 1):
        # ========== SAVE CHECKPOINT (last epoch + every save_freq epochs) ==========
        if epoch > args.train_params.start_epoch:
            if (
                args.train_params.save_freq
                and epoch % args.train_params.save_freq == 0
                or epoch == args.train_params.epochs
            ):
                save_model(
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    tag="last",
                    args=args,
                )
        
        # ========== TEST ONE EPOCH ==========
        new_best = False
        test_stats = {}
        if (
            args.train_params.eval_freq > 0
            and epoch % args.train_params.eval_freq == 0
            and epoch > 0
        ):
            for test_name, testset in data_loader_test.items():
                print(f"Testing on {test_name} ...")
                stats = test_one_epoch(
                    model,
                    teacher_extractor,
                    test_criterion,
                    testset,
                    device,
                    epoch,
                    args=args,
                    prefix=test_name,
                )
                test_stats[test_name] = stats

            # Calculate average test loss median
            avg_test_loss_med = np.mean(
                [stats["loss_med"] for stats in test_stats.values()]
            )
            test_stats["Average Test Loss Median"] = avg_test_loss_med
            # Save best
            if avg_test_loss_med < best_so_far:
                best_so_far = avg_test_loss_med
                new_best = True
    
        if epoch > args.train_params.start_epoch:
            if args.train_params.save_freq and epoch % args.train_params.save_freq == 0: # save every save_freq epochs
                print(f"[CHECKPOINT] Saving periodic checkpoint at epoch {epoch}")
                save_model(
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_loss=best_so_far,
                    tag=f"{epoch}",
                    args=args,
                )
            if new_best: # save if best
                print(f"[BEST] New best model found at epoch {epoch} with avg test loss median: {best_so_far:.6f}")
                save_model(
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_loss=best_so_far,
                    tag="best",
                    args=args,
                )
        if epoch >= args.train_params.epochs:
            break  # exit after writing last test to disk

        # ========== TRAIN ONE EPOCH ==========
        train_stats = train_one_epoch(
            model,
            teacher_extractor,
            train_criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args=args,
            param_groups_name_to_idx_map=param_groups_name_to_idx_map,
        )
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            scheduler.step()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
    pass

def build_dataset(
    dataset, num_workers, test, batch_size=None, max_num_of_imgs_per_gpu=None
):
    """
    Builds data loaders for training or testing.

    Args:
        dataset: Dataset specification string.
        num_workers: Number of worker processes for data loading.
        test: Boolean flag indicating whether this is a test dataset.
        batch_size: Number of samples per batch. Defaults to None. Used only for testing.
        max_num_of_imgs_per_gpu: Maximum number of images per GPU. Defaults to None. Used only for training.

    Returns:
        DataLoader: PyTorch DataLoader configured for the specified dataset.
    """
    split = ["Train", "Test"][test]
    print(f"Building {split} Data loader for dataset: ", dataset)
    if test:
        assert batch_size is not None, (
            "batch_size must be specified for testing dataloader"
        )
        loader = get_test_data_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_mem=True,
            shuffle=False,
            drop_last=False,
        )
    else:
        assert max_num_of_imgs_per_gpu is not None, (
            "max_num_of_imgs_per_gpu must be specified for training dataloader"
        )
        loader = get_train_data_loader(
            dataset=dataset,
            max_num_of_imgs_per_gpu=max_num_of_imgs_per_gpu,
            num_workers=num_workers,
            pin_mem=True,
            shuffle=True,
            drop_last=True,
        )

    print(f"{split} dataset length: ", len(loader))
    return loader

def train_one_epoch(
    model: torch.nn.Module,
    teacher_extractor: Optional[TeacherFeatureExtractor],
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    param_groups_name_to_idx_map=None,
):
    """
    Trains the model for one epoch.
    Epoch is just a chunk of the entire dataset.

    This function handles the training loop for a single epoch, including forward/backward passes,
    gradient accumulation, and logging metrics.

    Args:
        model: The neural network model to train.
        criterion: Loss function to optimize.
        data_loader: DataLoader providing the training data.
        optimizer: Optimizer for updating model parameters.
        device: Device to run training on (CPU or GPU).
        epoch: Current epoch number.
        args: Configuration object containing training parameters.
        param_groups_name_to_idx_map: Mapping from parameter group names to indices.

    Returns:
        dict: Dictionary containing training metrics averaged over the epoch.
    """
    model.train(True)
    metric_logger = train_tools.MetricLogger(delimiter=" | ")
    # NUOVO:
    header = f"Distillation Epoch: [{epoch}]"

    # Aggiungi meter per ogni LR di ogni submodule
    if param_groups_name_to_idx_map is not None:
        for submodule_name in param_groups_name_to_idx_map:
            lr_name = f"lr_{submodule_name}" if submodule_name != "default" else "lr"
            metric_logger.add_meter(
                lr_name, train_tools.SmoothedValue(window_size=1, fmt="{value:.6e}")
            )
    accum_iter = args.train_params.accum_iter

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)
    if hasattr(data_loader, "batch_sampler") and hasattr(
        data_loader.batch_sampler, "set_epoch"
    ):
        data_loader.batch_sampler.set_epoch(epoch)

    optimizer.zero_grad()

    # DINOv2 normalization stats
    DINOV2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    DINOV2_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.train_params.print_freq, header)
    ):
        n_views = len(batch)
        print(f"number of views in batch: {n_views}")

        # ========== PREPARE INPUT IMAGES FOR TEACHER ==========
        pil_images = []
        for view in batch:
            img_tensor = view.get("img")
            if img_tensor is None:
                continue

            # img_tensor è (N, 3, H, W)
            img_tensor = img_tensor * DINOV2_STD + DINOV2_MEAN
            img_tensor = torch.clamp(img_tensor, 0, 1) # check this

            for batch_idx in range(img_tensor.shape[0]):
                img_single = img_tensor[batch_idx]  # (3, H, W)
                img_np = (img_single.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                # pil_images.append(Image.fromarray(img_pil))
                img_pil = Image.fromarray(img_np)
                img_pil = img_pil.resize((1024, 1024), Image.Resampling.BILINEAR)
                pil_images.append(img_pil)

        # ========== EXTRACT TEACHER FEATURES ==========
        with torch.no_grad():
            teacher_features = teacher_extractor(pil_images).to(device, non_blocking=True)
        # ==============================================================================================
        if (args.train_params.save_pca_visualization_every is not None and (epoch % args.train_params.save_pca_visualization_every == 0 or epoch == 0)):
            save_pca_visualization_path = Path(os.path.join(args.output_dir, args.train_params.run_name))
        else:
            save_pca_visualization_path = None
        result = loss_of_one_batch_multi_view(
            batch,
            model,
            criterion,
            device,
            # use_amp=bool(args.train_params.amp),
            # amp_dtype=args.train_params.amp_dtype,
            # ret="loss",
            teacher_features=teacher_features,
            save_pca_visualization_path=save_pca_visualization_path,
            epoch=epoch,
        )
        loss, loss_details = result["loss"]

        if n_views > 2:
            loss = loss * (
                2 / n_views
            )  # scale the loss relative to the number of views (base is 2 views)
        loss_value = float(loss) # questo serve per il logging

        # Scale the loss by the number of gradient accumulation iterations
        loss /= accum_iter

        def grad_norm(module):
            total = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    total += p.grad.detach().float().norm(2).item() ** 2
            return total ** 0.5

        def has_nan(t):
            return torch.isnan(t).any().item() or torch.isinf(t).any().item()

        def _first_param(module):
            for p in module.parameters():
                return p
            return None

        # print("[BEFORE] dpt_feature_head_2:", grad_norm(model.dpt_feature_head_2))
        # print("[BEFORE] sam2_compat:", grad_norm(model.sam2_compat))

        # ================== DEBUG CHAT ==================
        # print("[BEFORE] Grad norms in dpt_feature_head_2:")
        # for name, p in model.dpt_feature_head_2.named_parameters():
        #     if p.grad is not None:
        #         print(name, p.grad.norm().item())
        # ================================================

        loss.backward()

        # ================== DEBUG CHAT ==================
        # print("[AFTER] Grad norms in dpt_feature_head_2:")
        # for name, p in model.dpt_feature_head_2.named_parameters():
        #     if p.grad is not None:
        #         print(name, p.grad.norm().item())
        # ================================================

        # print("[AFTER] dpt_feature_head_2:", grad_norm(model.dpt_feature_head_2))
        # print("[AFTER] sam2_compat:", grad_norm(model.sam2_compat))

        # for name, p in model.dpt_feature_head_2.named_parameters():
        #     if p.grad is not None and has_nan(p.grad):
        #         print("[NAN] grad in", name)
        #     if has_nan(p.data):
        #         print("[NAN] weight in", name)

        if (data_iter_step + 1) % accum_iter == 0:
            # print(f"[STEP] optimizer step @ iter {data_iter_step}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            # with torch.no_grad():
            #     w = _first_param(model.dpt_feature_head_2)
            #     if w is not None:
            #         w_mean_before = w.mean().item()
            #         w_std_before = w.std().item()
            #     else:
            #         w_mean_before = float("nan")
            #         w_std_before = float("nan")


            # with torch.no_grad():
            #     w = _first_param(model.dpt_feature_head_2)
            #     if w is not None:
            #         w_mean_after = w.mean().item()
            #         w_std_after = w.std().item()
            #     else:
            #         w_mean_after = float("nan")
            #         w_std_after = float("nan")

            # print(f"[HEAD] w_mean {w_mean_before:.6f} -> {w_mean_after:.6f} | w_std {w_std_before:.6f} -> {w_std_after:.6f}")

        del loss
        del batch

        metric_logger.update(loss=loss_value, **loss_details)

        # Aggiorna i LR per ogni submodule se disponibili i mapping
        if param_groups_name_to_idx_map is not None:
            for submodule_name in param_groups_name_to_idx_map:
                lr_name = f"lr_{submodule_name}" if submodule_name != "default" else "lr"
                # Prendi il primo indice del gruppo (contiene il LR corretto)
                group_idx = param_groups_name_to_idx_map[submodule_name][0]
                log_lr = optimizer.param_groups[group_idx]["lr"]
                metric_logger.update(**{lr_name: log_lr})

    # # Gather the stats from all processes
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    teacher_extractor: Optional[TeacherFeatureExtractor],
    criterion: torch.nn.Module,
    data_loader: Sized,
    device: torch.device,
    epoch: int,
    args,
    prefix="test",
):
    """
    Evaluates the model on a test dataset for one epoch.
    Epoch is just a chunk of the entire dataset.

    This function runs evaluation on the test dataset without computing gradients,
    and collects metrics for model performance assessment with teacher features.

    Args:
        model: The neural network model to evaluate.
        teacher_extractor: Teacher feature extractor for distillation.
        criterion: Loss function for evaluation.
        data_loader: DataLoader providing the test data.
        device: Device to run evaluation on (CPU or GPU).
        epoch: Current epoch number.
        args: Configuration object containing evaluation parameters.
        prefix: String prefix for logging metrics.

    Returns:
        dict: Dictionary containing evaluation metrics (average and median values).
    """
    from collections import defaultdict
    
    model.eval()
    metric_logger = train_tools.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(
        lambda: train_tools.SmoothedValue(window_size=9**9)
    )
    header = "Distillation Test Epoch: [{}]".format(epoch)


    # se true fa la validazione sempre sugli stessi samples
    if args.train_params.freeze_val_samples_across_all_epochs:
        dataloader_epoch = 0
    else:
        dataloader_epoch = epoch
    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(dataloader_epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(dataloader_epoch)
    if hasattr(data_loader, "batch_sampler") and hasattr(
        data_loader.batch_sampler, "set_epoch"
    ):
        data_loader.batch_sampler.set_epoch(dataloader_epoch)

    # DINOv2 normalization stats
    DINOV2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    DINOV2_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    for _, batch in enumerate(
        metric_logger.log_every(data_loader, args.train_params.print_freq, header)
    ):
        n_views = len(batch)

        # ========== PREPARE INPUT IMAGES FOR TEACHER ==========
        pil_images = []
        for view in batch:
            img_tensor = view.get("img")
            if img_tensor is None:
                continue

            # img_tensor è (N, 3, H, W)
            img_tensor = img_tensor * DINOV2_STD + DINOV2_MEAN
            img_tensor = torch.clamp(img_tensor, 0, 1)

            for batch_idx in range(img_tensor.shape[0]):
                img_single = img_tensor[batch_idx]  # (3, H, W)
                img_pil = (img_single.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_pil))

        # ========== EXTRACT TEACHER FEATURES ==========
        teacher_features = teacher_extractor(pil_images).to(device, non_blocking=True)
        # ==============================================================================================

        result = loss_of_one_batch_multi_view(
            batch,
            model,
            criterion,
            device,
            teacher_features=teacher_features,
        )
        loss_value, loss_details = result["loss"]

        if n_views > 2:
            loss_value = loss_value * (
                2 / n_views
            )  # scale the loss relative to the number of views (base is 2 views)
        metric_logger.update(loss=float(loss_value), **loss_details)

    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    return results