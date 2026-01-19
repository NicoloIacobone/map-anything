import torch

def _tensor_bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()

def summarize_checkpoint_bytes(path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    def walk(obj, prefix=""):
        total = 0
        if isinstance(obj, torch.Tensor):
            return _tensor_bytes(obj)
        if isinstance(obj, dict):
            for k, v in obj.items():
                total += walk(v, prefix + f".{k}")
            return total
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                total += walk(v, prefix + f"[{i}]")
            return total
        return 0

    total = 0
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            b = walk(v, prefix=str(k))
            total += b
            print(f"{k:30s}: {b/1024**2:10.1f} MB")
    print(f"{'TOTAL(tensors)':30s}: {total/1024**2:10.1f} MB")