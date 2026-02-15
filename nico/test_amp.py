import sys
import time
import torch

def select_amp_settings():
    amp_enabled = False
    amp_dtype = "fp32"
    scaler_enabled = False

    if torch.cuda.is_available():
        # Preferisci bf16 se supportato; altrimenti fp16
        if torch.cuda.is_bf16_supported():
            amp_enabled = True
            amp_dtype = "bf16"
            scaler_enabled = False  # bf16 in genere non richiede GradScaler
        else:
            amp_enabled = True
            amp_dtype = "fp16"
            scaler_enabled = True   # fp16 → GradScaler fortemente consigliato
    return amp_enabled, amp_dtype, scaler_enabled

def print_env_info():
    print("=== Torch / CUDA Info ===")
    print(f"torch.__version__: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        dev_idx = 0
        print(f"CUDA device: {torch.cuda.get_device_name(dev_idx)}")
        cap = torch.cuda.get_device_capability(dev_idx)
        print(f"Compute capability: {cap[0]}.{cap[1]}")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    else:
        print("No CUDA → only fp32 on CPU.")

def test_autocast(dtype_str):
    if not torch.cuda.is_available():
        print(f"[autocast {dtype_str}] skipped (no CUDA)")
        return
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype_str, torch.float32)
    try:
        a = torch.randn((4096, 4096), device="cuda")
        b = torch.randn((4096, 4096), device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", enabled=(dtype!=torch.float32), dtype=dtype):
            c = a @ b
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"[autocast {dtype_str}] ok, time={dt:.3f}s, c.dtype={c.dtype}")
    except Exception as e:
        print(f"[autocast {dtype_str}] FAILED: {e}")

def test_grad_scaler(dtype_str):
    if not torch.cuda.is_available():
        print(f"[GradScaler {dtype_str}] skipped (no CUDA)")
        return
    use_scaler = (dtype_str == "fp16")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype_str, torch.float32)
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 10),
    ).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    x = torch.randn((128, 1024), device="cuda")
    y = torch.randn((128, 10), device="cuda")

    try:
        with torch.autocast("cuda", enabled=(dtype!=torch.float32), dtype=dtype):
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        opt.zero_grad(set_to_none=True)
        print(f"[GradScaler {dtype_str}] ok (used_scaler={use_scaler})")
    except Exception as e:
        print(f"[GradScaler {dtype_str}] FAILED: {e}")

def main():
    print_env_info()
    amp_enabled, amp_dtype, scaler_enabled = select_amp_settings()
    print("\n=== Recommended Local Settings ===")
    print(f"use_amp: {amp_enabled}")
    print(f"amp_dtype: {amp_dtype}")
    print(f"enable_grad_scaler: {scaler_enabled}")

    print("\n=== Autocast Tests ===")
    # bf16 test only if CUDA & bf16 supported
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        test_autocast("bf16")
    else:
        print("[autocast bf16] not supported on this device")

    # fp16 test if CUDA
    if torch.cuda.is_available():
        test_autocast("fp16")
    else:
        print("[autocast fp16] not supported (no CUDA)")

    print("\n=== GradScaler Tests ===")
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        test_grad_scaler("bf16")
    else:
        print("[GradScaler bf16] not necessary or not supported")

    if torch.cuda.is_available():
        test_grad_scaler("fp16")
    else:
        print("[GradScaler fp16] not supported (no CUDA)")

if __name__ == "__main__":
    sys.exit(main())