#!/bin/bash
#
# Specify job name.
#SBATCH --job-name=4_rtx_4090
#
# Specify output file.
#SBATCH --output=mapanything_%j.log
#
# Specify error file.
#SBATCH --error=mapanything_%j.err
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify time limit.
#SBATCH --time=03:59:59
#
# Specify number of tasks.
#SBATCH --ntasks=1
#
# Specify number of CPU cores per task.
#SBATCH --cpus-per-task=16
#
# Specify memory limit per CPU core.
#SBATCH --mem-per-cpu=8192
#
# Specify number of required GPUs.
#SBATCH --gpus=rtx_4090:4
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

# Usa automaticamente tutte le GPU disponibili
torchrun --nproc_per_node=$NUM_GPUS distillation_new.py \
  --distributed \
  --use_wandb \
  --wandb_name "distillation_2" \
  --epochs 10 \
  --batch_size 8 \
  --num_workers 16 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
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
  --save_visualizations
#   --wandb_resume_id uqxp8h1i \
#   --output_dir /cluster/work/igp_psr/niacobone/distillation/output/distillation_1 \
#   --resume_ckpt /cluster/work/igp_psr/niacobone/distillation/output/distillation_1/checkpoints/checkpoint_epoch8.pth

echo "=== Job finished at $(date) ==="
start_time=${SLURM_JOB_START_TIME:-$(date +%s)}
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) (hh:mm:ss)"