#!/bin/bash
#
# Specify job name.
#SBATCH --job-name=SV_coco_unfrozen_3
#
# Specify output file.
#SBATCH --output=SV_coco_unfrozen_3_%j.log
#
# Specify error file.
#SBATCH --error=SV_coco_unfrozen_3_%j.err
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify time limit.
#SBATCH --time=72:00:00
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
#SBATCH --gpus=rtx_4090:2
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

# Rileva automaticamente il numero di GPU allocate da SLURM
# SLURM_GPUS_ON_NODE contiene il numero di GPU (es. "4")
# CUDA_VISIBLE_DEVICES contiene gli ID separati da virgola (es. "0,1,2,3")
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Conta le virgole + 1 per ottenere il numero di GPU
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
    # Fallback: usa nvidia-smi
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

echo "Detected $NUM_GPUS GPUs: $CUDA_VISIBLE_DEVICES"

# Usa automaticamente tutte le GPU disponibili
torchrun --nproc_per_node=$NUM_GPUS distillation.py \
  --distributed \
  --use_wandb \
  --num_workers 8 \
  --dataset coco2017 \
  --wandb_name "SV_coco_unfrozen_3" \
  --epochs 1000 \
  --lr 1e-3 \
  --batch_size 4 \
  --eval_freq 10 \
  --save_freq 10 \
  --print_freq 250 \
  --amp \
  --save_visualizations \
  --mse_type sample \
  --lr_encoder_scale 1.0 \
  --num_info_sharing_blocks_unfreeze 24 \
  --no_augmentation
#   --lr_encoder_scale 0.05
#   --num_info_sharing_blocks_unfreeze 24 \
#   --output_dir /cluster/work/igp_psr/niacobone/distillation/output/SV_ETH3D \
#   --resume_ckpt /cluster/work/igp_psr/niacobone/distillation/output/SV_ETH3D/checkpoints/checkpoint_epoch250.pth \
#   --wandb_resume_id n0h9ug8q

echo "=== Job finished at $(date) ==="
start_time=${SLURM_JOB_START_TIME:-$(date +%s)}
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) (hh:mm:ss)"