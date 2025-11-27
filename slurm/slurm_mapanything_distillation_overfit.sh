#!/bin/bash
#
# Specify job name.
#SBATCH --job-name=overfit_1
#
# Specify output file.
#SBATCH --output=overfit_1%j.log
#
# Specify error file.
#SBATCH --error=overfit_1%j.err
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify time limit.
#SBATCH --time=01:00:00
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

python distillation_overfit.py \
  --epochs 2000 \
  --batch_size 2 \
  --lr 1e-4 \
  --num_workers 0 \
  --print_freq 500 \
  --eval_freq 1 \
  --save_freq 1 \
  --save_visualizations \
  --amp \
  --use_wandb \
  --wandb_name "2img_2bs_2kE" \
  --disable_scheduler \
  --debug_max_train_images 2 \
  --weight_decay 0

echo "=== Job finished at $(date) ==="
start_time=${SLURM_JOB_START_TIME:-$(date +%s)}
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) (hh:mm:ss)"