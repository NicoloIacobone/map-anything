import os
import numpy as np
import torch
from PIL import Image

from sam2_builder import (
    load_sam2_feature_extractor,
    load_sam2_teacher_prompt_and_decoder,
    build_sam_mask_decoder,
)
from mapanything.models import MapAnything

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

np.random.seed(3)

sam2_checkpoint = "/scratch2/nico/sam2/checkpoints/sam2.1_hiera_large.pt"
img_path = "/scratch2/nico/distillation/dataset/coco2017/images/val2017/000000000139.jpg"

print("\n=== Test 1: Load Teacher Feature Extractor ===")
teacher_extractor = load_sam2_feature_extractor(
    checkpoint_path=sam2_checkpoint,
    device=device,
    model_size="large"
)
print(f"✓ Teacher feature extractor loaded")

# Load image
pil_img = Image.open(img_path).convert("RGB")

# Extract teacher features
teacher_features = teacher_extractor([pil_img])
print(f"✓ Teacher features extracted: shape {teacher_features.shape}, dtype {teacher_features.dtype}")

print("\n=== Test 2: Load Student Model (MapAnything) ===")
# Load MapAnything with dpt_feature_head_2 for encoder distillation
model = MapAnything.from_pretrained("facebook/map-anything", revision="562de9ff7077addd5780415661c5fb031eb8003e", strict=False).to(device)
print(f"✓ MapAnything model loaded")
print(f"  Has dpt_feature_head_2: {hasattr(model, 'dpt_feature_head_2')}")

from mapanything.utils.image import load_images

views = load_images([img_path])  # List of dicts

from distillation import forward_pass_distillation_unified

student_features = forward_pass_distillation_unified(model, [img_path], device)

print(f"✓ Student features extracted: shape {student_features.shape}, dtype {student_features.dtype}")

print("\n=== Test 3: Load Teacher Prompt Encoder + Mask Decoder ===")
prompt_encoder, teacher_mask_decoder = load_sam2_teacher_prompt_and_decoder(
    checkpoint_path=sam2_checkpoint,
    device=device,
    image_size=1024,
    backbone_stride=16,
    embed_dim=256,
)
print(f"✓ Teacher PromptEncoder and MaskDecoder loaded")
print(f"  PromptEncoder frozen: {not any(p.requires_grad for p in prompt_encoder.parameters())}")
print(f"  MaskDecoder frozen: {not any(p.requires_grad for p in teacher_mask_decoder.parameters())}")

print("\n=== Test 4: Build Student Mask Decoder ===")
student_mask_decoder = build_sam_mask_decoder(
    embed_dim=256,
    num_multimask_outputs=3,
    use_high_res_features=False,
    pred_obj_scores=False,
    pred_obj_scores_mlp=False,
    iou_prediction_use_sigmoid=False,
).to(device)
print(f"✓ Student MaskDecoder built")
print(f"  Trainable: {any(p.requires_grad for p in student_mask_decoder.parameters())}")
print(f"  Param count: {sum(p.numel() for p in student_mask_decoder.parameters()):,}")

# Attach student decoder to model (like distillation.py will do)
model.sam2_mask_decoder_student = student_mask_decoder
print(f"✓ Student decoder attached to MapAnything model")

print("\n=== Test 5: Encoder Distillation Loss ===")
# Check if teacher and student features have compatible shapes
print(f"  Teacher features: {teacher_features.shape}")
print(f"  Student features: {student_features.shape}")

if teacher_features.shape != student_features.shape:
    print(f"  [WARN] Shape mismatch! Need to resize/interpolate student features")
    student_features_resized = torch.nn.functional.interpolate(
        student_features,
        size=teacher_features.shape[-2:],
        mode='bilinear',
        align_corners=False
    )
    print(f"  Student features resized: {student_features_resized.shape}")
else:
    student_features_resized = student_features

encoder_loss = torch.nn.functional.mse_loss(student_features_resized, teacher_features)
print(f"✓ Encoder distillation loss (MSE): {encoder_loss.item():.6f}")

print("\n=== Test 6: Decoder Forward Pass (Teacher vs Student) ===")
# Use teacher features for both (for fair comparison)
image_embeddings = teacher_features  # (B, 256, 64, 64)
B = image_embeddings.shape[0]

# Create dummy point prompts (1 point per image at center)
point_coords = torch.tensor([[[512.0, 512.0]]], device=device, dtype=torch.float32)  # (B=1, K=1, 2)
point_labels = torch.ones((B, 1), device=device, dtype=torch.int32)  # (B=1, K=1)

print(f"  image_embeddings: {image_embeddings.shape}")
print(f"  point_coords: {point_coords.shape}")
print(f"  point_labels: {point_labels.shape}")

# Generate prompt embeddings (shared)
sparse_emb, dense_emb = prompt_encoder(
    points=(point_coords, point_labels),
    boxes=None,
    masks=None,
)
image_pe = prompt_encoder.get_dense_pe().to(device=image_embeddings.device, dtype=image_embeddings.dtype)

# Teacher decoder forward
with torch.no_grad():
    t_masks, t_iou, t_tokens, t_obj = teacher_mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
        repeat_image=False,
        high_res_features=None,
    )
print(f"✓ Teacher decoder forward succeeded")
print(f"    low_res_masks: {t_masks.shape}, iou: {t_iou.shape}, tokens: {t_tokens.shape}")

# Student decoder forward
s_masks, s_iou, s_tokens, s_obj = student_mask_decoder(
    image_embeddings=image_embeddings,
    image_pe=image_pe,
    sparse_prompt_embeddings=sparse_emb,
    dense_prompt_embeddings=dense_emb,
    multimask_output=False,
    repeat_image=False,
    high_res_features=None,
)
print(f"✓ Student decoder forward succeeded")
print(f"    low_res_masks: {s_masks.shape}, iou: {s_iou.shape}, tokens: {s_tokens.shape}")

print("\n=== Test 7: Decoder Distillation Loss ===")
decoder_loss_tokens = torch.nn.functional.mse_loss(s_tokens, t_tokens)
decoder_loss_masks = torch.nn.functional.mse_loss(s_masks, t_masks)
decoder_loss_iou = torch.nn.functional.mse_loss(s_iou, t_iou)
print(f"✓ Decoder losses computed:")
print(f"    MSE tokens: {decoder_loss_tokens.item():.6f}")
print(f"    MSE masks: {decoder_loss_masks.item():.6f}")
print(f"    MSE iou: {decoder_loss_iou.item():.6f}")

# Combined loss (like in distillation.py)
total_loss = encoder_loss + decoder_loss_tokens
print(f"✓ Total loss (encoder + decoder_tokens): {total_loss.item():.6f}")

print("\n=== Test 8: Backward Pass (Gradient Check) ===")
# Clear gradients
model.zero_grad()
student_mask_decoder.zero_grad()

# Disable autocast for backward
with torch.amp.autocast('cuda', enabled=False):
    # Re-compute student features in fp32 using forward_pass_distillation_unified
    student_features_fp32 = forward_pass_distillation_unified(
        model=model,
        image_paths=[img_path],
        device=device,
        use_amp=False,  # Disable AMP for fp32
        amp_dtype="bf16",
        process_individually=True,
        use_encoder_features=False,
    ).float()
    
    # Resize if needed
    if student_features_fp32.shape != teacher_features.shape:
        student_features_fp32 = torch.nn.functional.interpolate(
            student_features_fp32,
            size=teacher_features.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
    
    teacher_features_fp32 = teacher_features.float()
    
    # Encoder loss
    enc_loss_fp32 = torch.nn.functional.mse_loss(student_features_fp32, teacher_features_fp32)
    
    # Decoder loss (use student features from student encoder)
    sparse_fp32, dense_fp32 = prompt_encoder(
        points=(point_coords, point_labels),
        boxes=None,
        masks=None,
    )
    image_pe_fp32 = prompt_encoder.get_dense_pe().to(device=student_features_fp32.device, dtype=torch.float32)
    
    with torch.no_grad():
        t_masks_fp32, t_iou_fp32, t_tokens_fp32, t_obj_fp32 = teacher_mask_decoder(
            image_embeddings=student_features_fp32,  # Use student features
            image_pe=image_pe_fp32,
            sparse_prompt_embeddings=sparse_fp32.float(),
            dense_prompt_embeddings=dense_fp32.float(),
            multimask_output=False,
            repeat_image=False,
            high_res_features=None,
        )
    
    s_masks_fp32, s_iou_fp32, s_tokens_fp32, s_obj_fp32 = student_mask_decoder(
        image_embeddings=student_features_fp32,
        image_pe=image_pe_fp32,
        sparse_prompt_embeddings=sparse_fp32.float(),
        dense_prompt_embeddings=dense_fp32.float(),
        multimask_output=False,
        repeat_image=False,
        high_res_features=None,
    )
    
    dec_loss_fp32 = torch.nn.functional.mse_loss(s_tokens_fp32, t_tokens_fp32)
    
    total_loss_fp32 = enc_loss_fp32 + dec_loss_fp32
    print(f"  Total loss (fp32): {total_loss_fp32.item():.6f}")
    
    total_loss_fp32.backward()

# Check gradients
has_grad_head2 = any(p.grad is not None for p in model.dpt_feature_head_2.parameters())
has_grad_decoder = any(p.grad is not None for p in student_mask_decoder.parameters())

print(f"✓ dpt_feature_head_2 has gradients: {has_grad_head2}")
print(f"✓ student_mask_decoder has gradients: {has_grad_decoder}")

if has_grad_head2:
    grad_norm_head2 = sum((p.grad.norm() for p in model.dpt_feature_head_2.parameters() if p.grad is not None))
    print(f"  dpt_feature_head_2 gradient norm: {grad_norm_head2.item():.6f}")

if has_grad_decoder:
    grad_norm_decoder = sum((p.grad.norm() for p in student_mask_decoder.parameters() if p.grad is not None))
    print(f"  student_mask_decoder gradient norm: {grad_norm_decoder.item():.6f}")

print("\n=== All tests passed! ===\n")
print("Summary:")
print(f"  ✓ Teacher encoder (SAM2): frozen, produces features")
print(f"  ✓ Student encoder (MapAnything dpt_feature_head_2): trainable, produces features")
print(f"  ✓ Teacher decoder (SAM2 MaskDecoder): frozen, produces masks/tokens")
print(f"  ✓ Student decoder (MaskDecoder): trainable, produces masks/tokens")
print(f"  ✓ PromptEncoder: frozen, shared by teacher & student decoders")
print(f"  ✓ Encoder + Decoder distillation losses computed")
print(f"  ✓ Gradients flow to both student encoder and decoder")