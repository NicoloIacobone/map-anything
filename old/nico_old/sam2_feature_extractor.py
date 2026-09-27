import torch
import torch.nn as nn
from torchvision.transforms import Normalize, Resize, ToTensor


class SAM2FeatureExtractor(nn.Module):
    """Minimal wrapper to extract SAM2 backbone features for distillation."""
    
    def __init__(self, sam2_model, resolution=1024):
        super().__init__()
        # Estrai solo l'image encoder
        self.image_encoder = sam2_model.image_encoder
        self.resolution = resolution
        
        # Transforms minimali (copia da SAM2Transforms)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = nn.Sequential(
            Resize((resolution, resolution)),
            Normalize(self.mean, self.std),
        )
        
        # Freeze il modello teacher
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.image_encoder.eval()
    
    @torch.no_grad()
    def forward(self, images):
        """
        Args:
            images: List of PIL Images or numpy arrays (H,W,3) or tensor batch (B,3,H,W)
        
        Returns:
            vision_features: Tensor (B, C, H', W') - le feature SAM2
        """
        # Preprocessing
        if isinstance(images, list):
            img_batch = torch.stack([self.transforms(self.to_tensor(img)) for img in images])
        elif isinstance(images, torch.Tensor) and images.dim() == 4:
            # Già un batch (B,3,H,W)
            img_batch = images
        else:
            # Singola immagine
            img_batch = self.transforms(self.to_tensor(images)).unsqueeze(0)
        
        img_batch = img_batch.to(next(self.image_encoder.parameters()).device)
        
        # Forward backbone
        backbone_out = self.image_encoder(img_batch)
        
        # Ritorna solo le vision features (ultimo livello FPN)
        return backbone_out["vision_features"]  # (B, C, H', W')


def load_sam2_feature_extractor(checkpoint_path, config_path, device="cuda"):
    """Helper per caricare il feature extractor da checkpoint SAM2."""
    from sam2.build_sam import build_sam2
    
    sam2_model = build_sam2(
        config_file=config_path,
        ckpt_path=checkpoint_path,
        device=device,
        apply_postprocessing=False
    )
    
    return SAM2FeatureExtractor(sam2_model, resolution=1024).to(device)

# Example usage:
# from sam2_feature_extractor import load_sam2_feature_extractor

# # Setup
# teacher = load_sam2_feature_extractor(
#     checkpoint_path="path/to/sam2.1_hiera_large.pt",
#     config_path="path/to/sam2.1_hiera_l.yaml",
#     device="cuda"
# )

# # Durante training
# for batch in dataloader:
#     images = batch["images"]  # PIL o numpy array
    
#     # Estrai teacher features
#     with torch.no_grad():
#         teacher_features = teacher(images)  # (B, 256, 64, 64)
    
#     # Student forward
#     student_features = student_model(images)
    
#     # Distillation loss
#     loss = distillation_loss(student_features, teacher_features)
#     loss.backward()