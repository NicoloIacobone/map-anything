import torch
import torch.nn as nn
from torchvision.transforms import Normalize, Resize, ToTensor
from sam2_minimal.modeling.backbones.hieradet import Hiera
from sam2_minimal.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2_minimal.modeling.position_encoding import PositionEmbeddingSine
from sam2_minimal.modeling.sam.prompt_encoder import PromptEncoder
from sam2_minimal.modeling.sam.mask_decoder import MaskDecoder
from sam2_minimal.modeling.sam.transformer import TwoWayTransformer


def build_sam2_image_encoder(model_size="large"):
    """Build SAM2 image encoder without Hydra/YAML."""
    
    if model_size == "large":
        # Hiera-L backbone config
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
    
    # FPN Neck config
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
    
    # Full image encoder
    encoder = ImageEncoder(
        scalp=1,
        trunk=trunk,
        neck=neck,
    )
    
    return encoder


def build_sam2_prompt_encoder(
    image_size: int = 1024,
    backbone_stride: int = 16,
    embed_dim: int = 256,
    mask_in_chans: int = 16,
):
    """Build SAM2 PromptEncoder without Hydra/YAML."""
    image_embedding_size = (image_size // backbone_stride, image_size // backbone_stride)
    return PromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=image_embedding_size,
        input_image_size=(image_size, image_size),
        mask_in_chans=mask_in_chans,
    )


def build_sam2_mask_decoder(
    embed_dim: int = 256,
    use_high_res_features: bool = False,
    pred_obj_scores: bool = False,
    pred_obj_scores_mlp: bool = False,
    iou_prediction_use_sigmoid: bool = False,
):
    """Build SAM2 MaskDecoder without Hydra/YAML (matches SAM2Base._build_sam_heads)."""
    transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=embed_dim,
        num_heads=8,
        mlp_dim=2048,
    )
    return MaskDecoder(
        num_multimask_outputs=3,
        transformer=transformer,
        transformer_dim=embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        use_high_res_features=use_high_res_features,
        iou_prediction_use_sigmoid=iou_prediction_use_sigmoid,
        pred_obj_scores=pred_obj_scores,
        pred_obj_scores_mlp=pred_obj_scores_mlp,
    )


class SAM2FeatureExtractor(nn.Module):
    """Minimal wrapper to extract SAM2 backbone features."""
    
    def __init__(self, image_encoder, resolution=1024):
        super().__init__()
        self.image_encoder = image_encoder
        self.resolution = resolution
        
        # SAM2 preprocessing
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = nn.Sequential(
            Resize((resolution, resolution)),
            Normalize(self.mean, self.std),
        )
        
        # Freeze teacher
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.image_encoder.eval()
    
    @torch.no_grad()
    def forward(self, images):
        """
        Args:
            images: List of PIL/numpy images or tensor batch (B,3,H,W)
        Returns:
            vision_features: (B, 256, 64, 64)
        """
        # Preprocess
        if isinstance(images, list):
            img_batch = torch.stack([
                self.transforms(self.to_tensor(img)) for img in images
            ])
        elif isinstance(images, torch.Tensor):
            if images.dim() == 3:  # Single image (3,H,W)
                img_batch = self.transforms(images).unsqueeze(0)
            else:  # Batch (B,3,H,W)
                img_batch = images
        else:
            # Single PIL/numpy
            img_batch = self.transforms(self.to_tensor(images)).unsqueeze(0)
        
        img_batch = img_batch.to(next(self.image_encoder.parameters()).device)
        
        # Forward through encoder
        backbone_out = self.image_encoder(img_batch)
        return backbone_out["vision_features"]


class SAM2FullTeacher(nn.Module):
    """Teacher wrapper exposing both encoder features and SAM-style decoder outputs."""

    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        resolution: int = 1024,
        backbone_stride: int = 16,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.resolution = resolution
        self.backbone_stride = backbone_stride

        # SAM2 preprocessing
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = nn.Sequential(
            Resize((resolution, resolution)),
            Normalize(self.mean, self.std),
        )

        # Freeze teacher
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.mask_decoder.parameters():
            p.requires_grad = False

        self.image_encoder.eval()
        self.prompt_encoder.eval()
        self.mask_decoder.eval()

    def _preprocess(self, images):
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
        return img_batch

    def _empty_points(self, batch_size: int, device: torch.device):
        # Empty points; PromptEncoder will pad with a single "not a point" token.
        point_coords = torch.zeros((batch_size, 0, 2), device=device, dtype=torch.float32)
        point_labels = torch.zeros((batch_size, 0), device=device, dtype=torch.int32)
        return point_coords, point_labels

    def decode(
        self,
        image_embeddings: torch.Tensor,
        multimask_output: bool = False,
        high_res_features=None,
    ):
        """Run SAM prompt encoder + mask decoder on precomputed embeddings.

        Note: this function is intentionally NOT decorated with no_grad, so it can
        be used to backpropagate gradients to the input embeddings (student distillation).
        """
        b = image_embeddings.shape[0]
        device = image_embeddings.device
        point_coords, point_labels = self._empty_points(b, device)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_pred, sam_tokens, obj_scores = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        return {
            "low_res_masks": low_res_masks,
            "iou_pred": iou_pred,
            "sam_tokens": sam_tokens,
            "obj_scores": obj_scores,
        }

    @torch.no_grad()
    def forward(self, images, return_decoder: bool = False, multimask_output: bool = False):
        img_batch = self._preprocess(images)
        img_batch = img_batch.to(next(self.image_encoder.parameters()).device)

        backbone_out = self.image_encoder(img_batch)
        encoder_feats = backbone_out["vision_features"]

        out = {"encoder": encoder_feats}
        if return_decoder:
            out["decoder"] = self.decode(
                encoder_feats,
                multimask_output=multimask_output,
                high_res_features=None,
            )
        return out


class SAM2AMGModel(nn.Module):
    """A minimal SAM2 model wrapper compatible with SAM2ImagePredictor/AMG.

    This intentionally implements only what SAM2ImagePredictor needs:
    - forward_image
    - _prepare_backbone_features
    - sam_prompt_encoder / sam_mask_decoder attributes

    It avoids Hydra and avoids SAM2Base memory components.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        sam_prompt_encoder: nn.Module,
        sam_mask_decoder: nn.Module,
        image_size: int = 1024,
        backbone_stride: int = 16,
        use_high_res_features_in_sam: bool = False,
        directly_add_no_mem_embed: bool = False,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.sam_prompt_encoder = sam_prompt_encoder
        self.sam_mask_decoder = sam_mask_decoder

        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Present for predictor compatibility; not used when directly_add_no_mem_embed=False.
        self.no_mem_embed = torch.zeros(1, 1, getattr(self.image_encoder.neck, "d_model", 256))

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def forward_image(self, img_batch: torch.Tensor):
        """Match SAM2Base.forward_image()."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Match SAM2Base._prepare_backbone_features()."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes


def load_sam2_feature_extractor(checkpoint_path, device="cuda", model_size="large"):
    """
    Load SAM2 feature extractor from checkpoint.
    
    Args:
        checkpoint_path: Path to sam2.1_hiera_large.pt
        device: cuda/cpu
        model_size: "large" (add "base"/"small" if needed)
    
    Returns:
        SAM2FeatureExtractor module
    """
    # Build encoder architecture
    image_encoder = build_sam2_image_encoder(model_size)
    
    # Load pretrained weights
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract only image_encoder weights
    encoder_state = {}
    for k, v in checkpoint["model"].items():
        if k.startswith("image_encoder."):
            new_key = k.replace("image_encoder.", "")
            encoder_state[new_key] = v
    
    # Load weights
    missing, unexpected = image_encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"Warning: Missing keys: {missing[:5]}...")  # Show first 5
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected[:5]}...")
    
    # Wrap in feature extractor
    feature_extractor = SAM2FeatureExtractor(image_encoder, resolution=1024)
    return feature_extractor.to(device)


def load_sam2_full_teacher(
    checkpoint_path,
    device="cuda",
    model_size="large",
    resolution: int = 1024,
    backbone_stride: int = 16,
):
    """Load SAM2 full teacher (image_encoder + sam_prompt_encoder + sam_mask_decoder).

    Follows the same filtering approach used in load_sam2_feature_extractor.
    """
    image_encoder = build_sam2_image_encoder(model_size)
    prompt_encoder = build_sam2_prompt_encoder(
        image_size=resolution,
        backbone_stride=backbone_stride,
        embed_dim=256,
        mask_in_chans=16,
    )
    mask_decoder = build_sam2_mask_decoder(embed_dim=256)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    # image_encoder weights
    encoder_state = {}
    for k, v in state.items():
        if k.startswith("image_encoder."):
            encoder_state[k.replace("image_encoder.", "")] = v
    missing, unexpected = image_encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"Warning: Missing image_encoder keys: {missing[:5]}...")
    if unexpected:
        print(f"Warning: Unexpected image_encoder keys: {unexpected[:5]}...")

    # prompt encoder weights
    prompt_state = {}
    for k, v in state.items():
        if k.startswith("sam_prompt_encoder."):
            prompt_state[k.replace("sam_prompt_encoder.", "")] = v
    missing, unexpected = prompt_encoder.load_state_dict(prompt_state, strict=False)
    if missing:
        print(f"Warning: Missing sam_prompt_encoder keys: {missing[:5]}...")
    if unexpected:
        print(f"Warning: Unexpected sam_prompt_encoder keys: {unexpected[:5]}...")

    # mask decoder weights
    decoder_state = {}
    for k, v in state.items():
        if k.startswith("sam_mask_decoder."):
            decoder_state[k.replace("sam_mask_decoder.", "")] = v
    missing, unexpected = mask_decoder.load_state_dict(decoder_state, strict=False)
    if missing:
        print(f"Warning: Missing sam_mask_decoder keys: {missing[:5]}...")
    if unexpected:
        print(f"Warning: Unexpected sam_mask_decoder keys: {unexpected[:5]}...")

    teacher = SAM2FullTeacher(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        resolution=resolution,
        backbone_stride=backbone_stride,
    )
    return teacher.to(device)


def load_sam2_amg_model(
    checkpoint_path,
    device="cuda",
    model_size="large",
    resolution: int = 1024,
    backbone_stride: int = 16,
):
    """Load a minimal SAM2 model for SAM2ImagePredictor/SAM2AutomaticMaskGenerator."""
    image_encoder = build_sam2_image_encoder(model_size)
    prompt_encoder = build_sam2_prompt_encoder(
        image_size=resolution,
        backbone_stride=backbone_stride,
        embed_dim=256,
        mask_in_chans=16,
    )
    mask_decoder = build_sam2_mask_decoder(embed_dim=256)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    # image_encoder weights
    encoder_state = {}
    for k, v in state.items():
        if k.startswith("image_encoder."):
            encoder_state[k.replace("image_encoder.", "")] = v
    missing, unexpected = image_encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"Warning: Missing image_encoder keys: {missing[:5]}...")
    if unexpected:
        print(f"Warning: Unexpected image_encoder keys: {unexpected[:5]}...")

    # prompt encoder weights
    prompt_state = {}
    for k, v in state.items():
        if k.startswith("sam_prompt_encoder."):
            prompt_state[k.replace("sam_prompt_encoder.", "")] = v
    missing, unexpected = prompt_encoder.load_state_dict(prompt_state, strict=False)
    if missing:
        print(f"Warning: Missing sam_prompt_encoder keys: {missing[:5]}...")
    if unexpected:
        print(f"Warning: Unexpected sam_prompt_encoder keys: {unexpected[:5]}...")

    # mask decoder weights
    decoder_state = {}
    for k, v in state.items():
        if k.startswith("sam_mask_decoder."):
            decoder_state[k.replace("sam_mask_decoder.", "")] = v
    missing, unexpected = mask_decoder.load_state_dict(decoder_state, strict=False)
    if missing:
        print(f"Warning: Missing sam_mask_decoder keys: {missing[:5]}...")
    if unexpected:
        print(f"Warning: Unexpected sam_mask_decoder keys: {unexpected[:5]}...")

    model = SAM2AMGModel(
        image_encoder=image_encoder,
        sam_prompt_encoder=prompt_encoder,
        sam_mask_decoder=mask_decoder,
        image_size=resolution,
        backbone_stride=backbone_stride,
        use_high_res_features_in_sam=False,
        directly_add_no_mem_embed=False,
    )

    # Freeze (teacher)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    return model.to(device)