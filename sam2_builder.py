"""
SAM2 Components Builder

Custom constructor for SAM2 components (encoder, decoder, prompt encoder)
without Hydra/YAML dependencies. Centralizes all SAM2 module instantiation.

This module provides:
- build_sam2_image_encoder(): Construct SAM2 image encoder (teacher)
- build_sam_prompt_encoder(): Construct PromptEncoder for prompting
- build_sam_mask_decoder(): Construct MaskDecoder for decoding
- load_sam2_feature_extractor(): Load and wrap image encoder as inference module
- load_sam2_teacher_prompt_and_decoder(): Load teacher prompt+decoder from checkpoint
- SAM2FeatureExtractor: Inference wrapper for image encoder features
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torchvision.transforms import Normalize, Resize, ToTensor

from sam2_minimal.modeling.backbones.hieradet import Hiera
from sam2_minimal.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2_minimal.modeling.position_encoding import PositionEmbeddingSine
from sam2_minimal.modeling.sam.prompt_encoder import PromptEncoder
from sam2_minimal.modeling.sam.mask_decoder import MaskDecoder
from sam2_minimal.modeling.sam.transformer import TwoWayTransformer


# ==================== Image Encoder Builder ====================
def build_sam2_image_encoder(model_size: str = "large") -> ImageEncoder:
    """Build SAM2 image encoder without Hydra/YAML."""
    
    if model_size == "large":
        trunk = Hiera(
            embed_dim=144,
            num_heads=2,
            stages=(2, 6, 36, 4),
            global_att_blocks=(23, 33, 43),
            window_pos_embed_bkg_spatial_size=(7, 7),
            window_spec=[8, 4, 16, 8],
        )
        backbone_channel_list = [1152, 576, 288, 144]
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
    )
    
    neck = FpnNeck(
        position_encoding=position_encoding,
        d_model=256,
        backbone_channel_list=backbone_channel_list,
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest",
    )
    
    encoder = ImageEncoder(
        scalp=1,
        trunk=trunk,
        neck=neck,
    )
    
    return encoder


# ==================== Prompt Encoder Builder ====================
def build_sam_prompt_encoder(
    image_size: int = 1024,
    backbone_stride: int = 16,
    embed_dim: int = 256,
    mask_in_chans: int = 16,
) -> PromptEncoder:
    """Build SAM2 PromptEncoder for converting prompts to embeddings."""
    image_embedding_size = (image_size // backbone_stride, image_size // backbone_stride)
    return PromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=image_embedding_size,
        input_image_size=(image_size, image_size),
        mask_in_chans=mask_in_chans,
    )


# ==================== Mask Decoder Builder ====================
def build_sam_mask_decoder(
    embed_dim: int = 256,
    num_multimask_outputs: int = 3,
    use_high_res_features: bool = False,
    pred_obj_scores: bool = False,
    pred_obj_scores_mlp: bool = False,
    iou_prediction_use_sigmoid: bool = False,
) -> MaskDecoder:
    """Build SAM2 MaskDecoder for converting embeddings+prompts to masks."""
    transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=embed_dim,
        num_heads=8,
        mlp_dim=2048,
    )
    return MaskDecoder(
        num_multimask_outputs=num_multimask_outputs,
        transformer=transformer,
        transformer_dim=embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_high_res_features=use_high_res_features,
        pred_obj_scores=pred_obj_scores,
        pred_obj_scores_mlp=pred_obj_scores_mlp,
        iou_prediction_use_sigmoid=iou_prediction_use_sigmoid,
    )


# ==================== Checkpoint Loading Utilities ====================
def _infer_teacher_decoder_flags_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, bool]:
    """Infer MaskDecoder config flags from checkpoint (conservative inference)."""
    keys = list(state_dict.keys())
    has_obj = any("object_score" in k or "obj_score" in k for k in keys)
    has_obj_mlp = any("pred_obj_scores_mlp" in k or "object_score_mlp" in k for k in keys)
    # has_obj_mlp = any("pred_obj_score_head.layers" in k for k in keys)  # ← Fix: cerca .layers
    # has_high_res = any("conv_s0" in k or "conv_s1" in k for k in keys)  # ← Fix: rileva conv
    return {
        "pred_obj_scores": has_obj,
        "pred_obj_scores_mlp": has_obj_mlp,
        "iou_prediction_use_sigmoid": False,
        "use_high_res_features": False,
    }


def load_sam2_teacher_prompt_and_decoder(
    checkpoint_path: str,
    device: torch.device,
    image_size: int = 1024,
    backbone_stride: int = 16,
    embed_dim: int = 256,
) -> Tuple[PromptEncoder, MaskDecoder]:
    """Load SAM2 teacher PromptEncoder and MaskDecoder from checkpoint (both frozen)."""
    print(f"[INFO] Loading SAM2 checkpoint for teacher PromptEncoder and MaskDecoder")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt.get("model", ckpt)

    flags = _infer_teacher_decoder_flags_from_state_dict(sd)

    prompt_encoder = build_sam_prompt_encoder(
        image_size=image_size,
        backbone_stride=backbone_stride,
        embed_dim=embed_dim,
        mask_in_chans=16,
    )
    mask_decoder = build_sam_mask_decoder(
        embed_dim=embed_dim,
        num_multimask_outputs=3,
        use_high_res_features=flags["use_high_res_features"],
        pred_obj_scores=flags["pred_obj_scores"],
        pred_obj_scores_mlp=flags["pred_obj_scores_mlp"],
        iou_prediction_use_sigmoid=flags["iou_prediction_use_sigmoid"],
    )

    pe_sd = {
        k.replace("sam_prompt_encoder.", ""): v
        for k, v in sd.items()
        if k.startswith("sam_prompt_encoder.")
    }
    md_sd = {
        k.replace("sam_mask_decoder.", ""): v
        for k, v in sd.items()
        if k.startswith("sam_mask_decoder.")
    }

    missing_pe, unexpected_pe = prompt_encoder.load_state_dict(pe_sd, strict=False)
    missing_md, unexpected_md = mask_decoder.load_state_dict(md_sd, strict=False)
    
    if missing_pe or unexpected_pe:
        print(f"[WARN] PromptEncoder: missing={len(missing_pe)}, unexpected={len(unexpected_pe)}")
    if missing_md or unexpected_md:
        print(f"[WARN] MaskDecoder: missing={len(missing_md)}, unexpected={len(unexpected_md)}")
        if missing_md:
            print(f"  Missing keys: {missing_md[:5]}")  # Prime 5
        if unexpected_md:
            print(f"  Unexpected keys: {unexpected_md[:5]}")  # Prime 5

    prompt_encoder = prompt_encoder.to(device).eval()
    mask_decoder = mask_decoder.to(device).eval()
    
    for p in prompt_encoder.parameters():
        p.requires_grad = False
    for p in mask_decoder.parameters():
        p.requires_grad = False

    print("[INFO] SAM2 teacher PromptEncoder and MaskDecoder loaded and frozen")
    return prompt_encoder, mask_decoder


# ==================== Feature Extractor ====================
class SAM2FeatureExtractor(nn.Module):
    """Minimal wrapper to extract SAM2 image encoder features."""
    
    def __init__(self, image_encoder: ImageEncoder, resolution: int = 1024):
        super().__init__()
        self.image_encoder = image_encoder
        self.resolution = resolution
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = nn.Sequential(
            Resize((resolution, resolution)),
            Normalize(self.mean, self.std),
        )
        
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.image_encoder.eval()
    
    @torch.no_grad()
    def forward(self, images):
        """Extract image encoder features."""
        if isinstance(images, list):
            img_batch = torch.stack([
                self.transforms(self.to_tensor(img)) for img in images
            ])
        elif isinstance(images, torch.Tensor):
            if images.dim() == 3:
                img_batch = self.transforms(images).unsqueeze(0)
            else:
                img_batch = images
        else:
            img_batch = self.transforms(self.to_tensor(images)).unsqueeze(0)
        
        img_batch = img_batch.to(next(self.image_encoder.parameters()).device)
        backbone_out = self.image_encoder(img_batch)
        return backbone_out["vision_features"]


def load_sam2_feature_extractor(
    checkpoint_path: str,
    device: str = "cuda",
    model_size: str = "large",
) -> SAM2FeatureExtractor:
    """Load SAM2 feature extractor from checkpoint."""
    image_encoder = build_sam2_image_encoder(model_size)
    
    print(f"[INFO] Loading SAM2 checkpoint for teacher feature extractor")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    encoder_state = {}
    for k, v in checkpoint["model"].items():
        if k.startswith("image_encoder."):
            new_key = k.replace("image_encoder.", "")
            encoder_state[new_key] = v
    
    missing, unexpected = image_encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"[WARN] Image encoder missing: {len(missing)} keys (first 5:)")
        for k in list(missing)[:5]:
            print(f"      {k}")
    if unexpected:
        print(f"[WARN] Image encoder unexpected: {len(unexpected)} keys (first 5:)")
        for k in list(unexpected)[:5]:
            print(f"      {k}")
    
    feature_extractor = SAM2FeatureExtractor(image_encoder, resolution=1024)
    return feature_extractor.to(device)