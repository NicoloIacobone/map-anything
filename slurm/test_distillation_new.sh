#!/bin/bash
#
# Specify job name.
#SBATCH --job-name=test_decoder_distillation
#
# Specify output file.
#SBATCH --output=test_decoder_distillation_%j.log
#
# Specify error file.
#SBATCH --error=test_decoder_distillation_%j.err
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify time limit.
#SBATCH --time=00:10:00
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
#SBATCH --gpus=rtx_4090:1
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
python distillation.py --epochs 5 --num_workers 1 --debug_max_train_images 5 --debug_max_val_images 5 --print_freq 1 --log_freq 1 --lr 1e-3 --lr_encoder_scale 1.0 --lr_decoder_scale 0.5 --lr_transformer_scale 0.1 --lr_dino_scale 0.01 --mse_weight 0.6 --cosine_weight 0.4 --decoder_masks_weight 0.5 --decoder_iou_weight 0.3 --decoder_tokens_weight 0.2 --num_info_sharing_blocks_unfreeze 12 --num_dino_layers_unfreeze 24 --save_freq 1

echo "=== Job finished at $(date) ==="
start_time=${SLURM_JOB_START_TIME:-$(date +%s)}
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) (hh:mm:ss)"