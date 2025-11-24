class SAM2CompatibilityLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=1e-6)
        self.proj = nn.Conv2d(channels, 256, kernel_size=1)

    def forward(self, x):  # x: (B,C,H,W)
        # LayerNorm expects (B,H*W,C) or (B,C)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)        # (B,H,W,C)
        x = self.ln(x)                  # LN over C
        x = x.permute(0, 3, 1, 2)       # (B,C,H,W)
        x = self.proj(x)                # match SAM2 feature dim
        return x