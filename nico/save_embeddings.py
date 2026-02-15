import os
import sys
import time
from pathlib import Path

import torch

from mapanything.models import MapAnything
from mapanything.utils.image import load_images


# ==================== CONSTANTS (fill these) ====================
# Model to load from Hugging Face Hub or local
MODEL_NAME = "facebook/map-anything"

# Path to a checkpoint saved by distillation that contains only dpt_feature_head_2
# e.g., "/scratch2/nico/distillation/output/<run_name>/checkpoints/checkpoint_best.pth"
CHECKPOINT_PATH = "/scratch2/nico/distillation/output/distillation_4/checkpoints/checkpoint_epoch9.pth"

# Single image path to run inference on
IMAGE_PATH = "/scratch2/nico/distillation/dataset/coco2017/images/val2017/000000002587.jpg"

# Where to save the student embeddings tensor (.pt)
OUTPUT_PATH = "/scratch2/nico/distillation/output/distillation_4/visualizations/student/000000002587.pt"

# AMP settings (mirrors distillation.py behavior)
USE_AMP = True
AMP_DTYPE = "bf16"  # "bf16" or "fp16"


def main():
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 1) Instantiate model (strict=False to allow extra head)
    print("[INFO] Loading MapAnything model...")
    model = MapAnything.from_pretrained(MODEL_NAME, strict=False).to(device)

    model.eval()

    # 2) Load head-2 weights from checkpoint (as saved by distillation)
    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if not hasattr(model, "dpt_feature_head_2"):
        raise AttributeError("Model has no attribute 'dpt_feature_head_2' (second DPT head).")
    if "dpt_feature_head_2" not in ckpt:
        raise KeyError("Checkpoint does not contain 'dpt_feature_head_2' state.")
    model.dpt_feature_head_2.load_state_dict(ckpt["dpt_feature_head_2"], strict=True)
    print("[INFO] Loaded dpt_feature_head_2 from checkpoint.")

    # Load sam2_compat
    if hasattr(model, "sam2_compat"):
        if "sam2_compat" in ckpt:
            model.sam2_compat.load_state_dict(ckpt["sam2_compat"], strict=True)
            print("[INFO] Loaded sam2_compat from checkpoint.")
        else:
            print("[WARN] sam2_compat exists on model but not found in checkpoint. Using random initialization.")
    else:
        print("[WARN] Model does not have sam2_compat attribute. Check if enable_second_dense_head is active in config.")

    # 3) Load image via project utility (consistent with distillation.py)
    img_path = Path(IMAGE_PATH)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    views = load_images([str(img_path)])
    if len(views) == 0 or "img" not in views[0]:
        raise RuntimeError("Failed to load image with load_images().")

    # Match encoder dtype to avoid bf16/float mismatches (as seen during inference)
    try:
        enc_dtype = next(model.encoder.parameters()).dtype
    except Exception:
        # Fallback to float32 if encoder isn't accessible
        enc_dtype = torch.float32
    for v in views:
        if isinstance(v.get("img"), torch.Tensor):
            v["img"] = v["img"].to(device=device, dtype=enc_dtype, non_blocking=True)

    # 4) Run model forward (AMP settings aligned with distillation.py)
    if AMP_DTYPE == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16
    autocast_enabled = USE_AMP and (device.type == "cuda")

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=autocast_enabled, dtype=autocast_dtype):
        _ = model(
            views,
            memory_efficient_inference=False,
        )

    # 5) Fetch student embeddings from the added DPT head (as in distillation.py)
    base_model = model.module if hasattr(model, "module") else model
    student_embeddings = getattr(base_model, "_last_feat2_8x", None)
    if student_embeddings is None:
        raise KeyError("_last_feat2_8x not populated by the model forward. Check head integration.")
    print(f"[INFO] Student embeddings shape: {student_embeddings.shape}")
    # Save to disk (CPU tensor)
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student_embeddings.detach().cpu(), out_path)
    print(f"[SAVE] Student embeddings saved to: {out_path}")
    print(f"[DONE] Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERR] {e}")
        sys.exit(1)
    sys.exit(0)


    # ========== DEBUG: VERIFY MODEL STRUCTURE ==========
    # print("\n" + "="*70)
    # print("DEBUG: MODEL STRUCTURE VERIFICATION")
    # print("="*70)
    
    # # 1. Check if dpt_feature_head_2 exists
    # if hasattr(model, "dpt_feature_head_2"):
    #     print("✅ [OK] dpt_feature_head_2 exists on model")
    #     num_params = sum(p.numel() for p in model.dpt_feature_head_2.parameters())
    #     print(f"     → {num_params:,} parameters")
    # else:
    #     print("❌ [ERROR] dpt_feature_head_2 NOT FOUND on model")
    #     print("     → Check config.json: 'enable_second_dense_head' must be true")
    
    # # 2. Check if sam2_compat exists
    # if hasattr(model, "sam2_compat"):
    #     print("✅ [OK] sam2_compat exists on model")
    #     num_params = sum(p.numel() for p in model.sam2_compat.parameters())
    #     print(f"     → {num_params:,} parameters")
        
    #     # List parameters
    #     print("     → Parameters:")
    #     for name, param in model.sam2_compat.named_parameters():
    #         print(f"       - {name}: shape={tuple(param.shape)}")
    # else:
    #     print("❌ [ERROR] sam2_compat NOT FOUND on model")
    #     print("     → Check if SAM2CompatibilityLayer is initialized in model.py")
    
    # # 3. List all top-level modules
    # print("\n📦 Top-level modules:")
    # for name, _ in model.named_children():
    #     print(f"   - {name}")
    
    # # 4. Check parameters containing 'dpt_feature_head_2' or 'sam2_compat'
    # print("\n🔍 Parameters matching 'dpt_feature_head_2' or 'sam2_compat':")
    # found_head2 = False
    # found_sam2 = False
    # for name, param in model.named_parameters():
    #     if "dpt_feature_head_2" in name:
    #         if not found_head2:
    #             print(f"   ✅ Found dpt_feature_head_2 parameters:")
    #             found_head2 = True
    #         print(f"      - {name}: shape={tuple(param.shape)}")
    #     elif "sam2_compat" in name:
    #         if not found_sam2:
    #             print(f"   ✅ Found sam2_compat parameters:")
    #             found_sam2 = True
    #         print(f"      - {name}: shape={tuple(param.shape)}")
    
    # if not found_head2:
    #     print("   ❌ No dpt_feature_head_2 parameters found")
    # if not found_sam2:
    #     print("   ❌ No sam2_compat parameters found")
    
    # print("="*70 + "\n")
    # ========== END DEBUG ==========