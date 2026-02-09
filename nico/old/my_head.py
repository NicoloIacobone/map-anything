import torch
import torch.nn as nn
import torch.nn.functional as F

class InstanceSegmentationHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=512, out_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((64, 64))
        )

    def forward(self, x):
        return self.head(x)

# class InstanceSegmentationHead(nn.Module):
#     def __init__(self, in_dim=256, out_dim=256):
#         super().__init__()
#         self.head = nn.Sequential(
#             nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
#             nn.BatchNorm2d(out_dim),
#             nn.ReLU(),
#             nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1),
#             nn.BatchNorm2d(out_dim),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((64, 64))
#         )

#     def forward(self, x):  # x = [B, 256, H, W]
#         return self.head(x)

# class InstanceSegmentationHead(nn.Module):
#     def __init__(self, in_dim=256, out_dim=256):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=1)
#         self.down = nn.AdaptiveAvgPool2d((64, 64))

#     def forward(self, x):  # x = [B, 256, 192, 296]
#         x = self.conv1(x)  # x = [B, 256, 192, 296]
#         x = self.down(x)   # x = [B, 256, 64, 64]
#         return x

# class InstanceSegmentationHead(nn.Module):
#     def __init__(self, in_dim=256, hidden_dim=256, emb_dim=64, use_fg_logit=True):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, emb_dim)
#         )
#         self.use_fg_logit = use_fg_logit
#         if use_fg_logit:
#             self.fg_head = nn.Sequential(
#                 nn.Linear(in_dim, hidden_dim//2),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim//2, 1)
#             )

#     def forward(self, x):
#         # x: [N_points, in_dim]
#         emb = self.proj(x)               # [N, emb_dim]
#         emb = F.normalize(emb, dim=-1)   # normalizza per contrastive / clustering
#         fg_logit = None
#         if self.use_fg_logit:
#             fg_logit = self.fg_head(x).squeeze(-1)  # [N]
#         return emb, fg_logit