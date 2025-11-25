#!/bin/bash
#
# Specify job name.
#SBATCH --job-name=2_rtx_4090
#
# Specify output file.
#SBATCH --output=unfreeze_4_%j.log
#
# Specify error file.
#SBATCH --error=unfreeze_4_%j.err
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify time limit.
#SBATCH --time=02:00:00
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

# # check if the dataset is available in /cluster/scratch/niacobone/distillation/coco2017
# if [ ! -d "/cluster/scratch/niacobone/distillation/coco2017" ]; then
#     echo "Dataset not found in /cluster/scratch/niacobone/distillation/coco2017 - copyting from /cluster/work/igp_psr/niacobone/coco2017"
#     cp -r /cluster/work/igp_psr/niacobone/coco2017 /cluster/scratch/niacobone/distillation/
#     echo "Dataset copied."
# fi

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

python distillation.py \
  --use_wandb \
  --wandb_name "test_6_unfreeze_4" \
  --epochs 50 \
  --batch_size 20 \
  --num_workers 8 \
  --lr 1e-4 \
  --weight_decay 0.0 \
  --lr_min 1e-6 \
  --clip_grad 1.0 \
  --accum_iter 1 \
  --mse_weight 0.5 \
  --cosine_weight 0.5 \
  --eval_freq 1 \
  --save_freq 1 \
  --print_freq 50 \
  --amp \
  --amp_dtype bf16 \
  --seed 42 \
  --save_visualizations \
  --disable_scheduler \
  --debug_max_train_images 4000 \
  --debug_max_val_images 1000 \
  --num_info_sharing_blocks_unfreeze 4  

echo "=== Job finished at $(date) ==="
start_time=${SLURM_JOB_START_TIME:-$(date +%s)}
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) (hh:mm:ss)"