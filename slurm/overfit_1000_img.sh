#!/bin/bash
#
# Specify job name.
#SBATCH --job-name=overfit_1000_img
#
# Specify output file.
#SBATCH --output=overfit_1000_img_%j.log
#
# Specify error file.
#SBATCH --error=overfit_1000_img_%j.err
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify time limit.
#SBATCH --time=48:00:00
#
# Specify number of tasks.
#SBATCH --ntasks=1
#
# Specify number of CPU cores per task.
#SBATCH --cpus-per-task=8
#
# Specify memory limit per CPU core.
#SBATCH --mem-per-cpu=4096
#
# Specify number of required GPUs.
#SBATCH --gpus=rtx_4090
#
# Specify disk limit on local scratch.
#SBATCH --tmp=500000
#
# Specify email notifications.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

echo "=== Job starting on $(hostname) at $(date) ==="
# DATE_VAR=$(date +%Y%m%d%H%M%S)

# Load modules.
module load stack/2024-06 python/3.12 cuda/12.4 eth_proxy
echo "Loaded modules: $(module list 2>&1)"

# Activate virtual environment for sam2.
source /cluster/scratch/niacobone/map-anything/myenv/bin/activate
echo "Activated Python venv: $(which python)"

# Execute
cd /cluster/scratch/niacobone/map-anything
echo "Starting MapAnything distillation..."

export WANDB_API_KEY=$(cat "/cluster/home/niacobone/.config/wandb/wandb_api_key.txt")

# Usa automaticamente tutte le GPU disponibili
python distillation_backup.py \
  --use_wandb \
  --wandb_name "overfit_1000_img" \
  --dataset coco2017 \
  --lr 2.5e-4 \
  --batch_size 8 \
  --num_workers 8 \
  --epochs 5000 \
  --debug_max_train_images 1000 \
  --debug_max_val_images 1 \
  --save_freq 500 \
  --save_visualizations_encoder \
  --amp \
  --print_freq 100 \
  --save_encoder_ckpt \
  --overfit

echo "=== Job finished at $(date) ==="
start_time=${SLURM_JOB_START_TIME:-$(date +%s)}
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) (hh:mm:ss)"