import itertools
import subprocess
import sys

# Lista dei parametri da esplorare
epochs_list = [5000]
lr_list = [1e-3, 5e-4, 1e-4]
norm_list = [False, True]

# Crea tutte le combinazioni possibili
grid = list(itertools.product(epochs_list, lr_list, norm_list))

print(f"Running {len(grid)} experiments in total...")

for i, (epochs, lr, norm) in enumerate(grid):
    print(f"\n=== [Run {i+1}/{len(grid)}] epochs={epochs}, lr={lr}, norm={norm} ===")

    # Nome univoco per WandB e per i file di log
    run_name = f"ep{epochs}_lr{lr}_norm{norm}".replace('.', '')
    log_file = f"logs/{run_name}.log"
    err_file = f"logs/{run_name}.err"

    # Costruisci il comando da eseguire
    cmd = [
        sys.executable,  # usa lo stesso interprete Python
        "distillation_overfit.py",
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--wandb_name", run_name
    ]
    if norm:
        cmd.append("--norm")

    # Esegui e reindirizza log
    # with open(log_file, "w") as out, open(err_file, "w") as err:
    #     subprocess.run(cmd, stdout=out, stderr=err)
    subprocess.run(cmd)

print("\n>>> All experiments completed.")