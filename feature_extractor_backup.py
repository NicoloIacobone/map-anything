import torch
import torch.nn as nn
from torchvision.transforms import Normalize, Resize, ToTensor
from sam2_minimal.modeling.backbones.hieradet import Hiera
from sam2_minimal.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2_minimal.modeling.position_encoding import PositionEmbeddingSine


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