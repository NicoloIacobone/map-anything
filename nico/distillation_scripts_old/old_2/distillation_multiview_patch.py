# ====================================================================================
# PATCH PER SUPPORTO MULTI-VIEW in distillation.py
# ====================================================================================
# 
# Inserire questa funzione dopo forward_pass_distillation() (circa linea 500):

def forward_pass_multiview_distillation(
    model: torch.nn.Module,
    batch: Dict,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: str = "bf16",
) -> torch.Tensor:
    """
    Esegue la forward di MapAnything per estrarre le feature dello studente in multi-view mode.
    Processa ogni scena (gruppo di view) separatamente, poi concatena i risultati.
    
    Args:
        model: MapAnything model
        batch: Dict with 'image_paths' (flat list), 'num_views_per_scene' (list of ints)
        device: Device to run on
        use_amp: Whether to use automatic mixed precision
        amp_dtype: AMP dtype ("bf16" or "fp16")
    
    Returns:
        student_features: (B*N, C, H, W) tensor where B=num_scenes, N=views_per_scene
    """
    from mapanything.utils.image import load_images
    
    image_paths = batch["image_paths"]
    num_views_per_scene = batch["num_views_per_scene"]
    
    if amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16
    autocast_enabled = use_amp and (device.type == "cuda")
    
    # Raggruppa i path per scena
    all_student_feats = []
    start_idx = 0
    for n_views in num_views_per_scene:
        scene_paths = image_paths[start_idx:start_idx + n_views]
        scene_views = load_images(scene_paths)
        
        for v in scene_views:
            img = v.get("img")
            if isinstance(img, torch.Tensor):
                v["img"] = img.to(device, non_blocking=True)
        
        # Forward per questa scena (multi-view transformer lavora internamente)
        with torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=autocast_dtype):
            _ = model(scene_views, memory_efficient_inference=False)
        
        base_model = model.module if hasattr(model, "module") else model
        student_feats = getattr(base_model, "_last_feat2_8x", None)
        if student_feats is None:
            raise KeyError("Student features not found (_last_feat2_8x)")
        
        # student_feats shape: (n_views, C, H, W)
        all_student_feats.append(student_feats)
        start_idx += n_views
    
    # Concatena tutte le feature: (B*N, C, H, W)
    import torch
    return torch.cat(all_student_feats, dim=0)


# ====================================================================================
# ESEMPIO FILE JSON PER SCENES:
# ====================================================================================
# 
# train_scenes.json:
# {
#   "scene_001": [
#     "/path/to/train2017/img_001_view0.jpg",
#     "/path/to/train2017/img_001_view1.jpg",
#     "/path/to/train2017/img_001_view2.jpg"
#   ],
#   "scene_002": [
#     "/path/to/train2017/img_002_view0.jpg",
#     "/path/to/train2017/img_002_view1.jpg"
#   ],
#   ...
# }
#
# ====================================================================================
# COMANDO DI ESEMPIO PER USARE MULTI-VIEW MODE:
# ====================================================================================
#
# torchrun --nproc_per_node=4 distillation.py \
#   --distributed \
#   --use_wandb \
#   --wandb_project "mapanything-multiview-distillation" \
#   --wandb_name "multiview_run_1" \
#   --multi_view_mode \
#   --train_scenes_file /path/to/train_scenes.json \
#   --val_scenes_file /path/to/val_scenes.json \
#   --epochs 100 \
#   --batch_size 2 \
#   --num_workers 8 \
#   --lr 1e-4 \
#   --amp \
#   --amp_dtype bf16
#
# Note:
# - batch_size in multi-view mode è il numero di SCENE per batch (non immagini totali)
# - Se ogni scena ha 3 view e batch_size=2, ogni forward processerà 6 immagini totali
# - Le teacher features devono essere pre-computate per ogni singola view
