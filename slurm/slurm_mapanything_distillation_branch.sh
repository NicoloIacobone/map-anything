#!/bin/bash
#
# Specify job name.
#SBATCH --job-name=mapanything_distillation
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
#SBATCH --time=23:59:59
#
# Specify number of tasks.
#SBATCH --ntasks=1
#
# Specify number of CPU cores per task.
#SBATCH --cpus-per-task=1
#
# Specify memory limit per CPU core.
#SBATCH --mem-per-cpu=8192
#
# Specify number of required GPUs.
#SBATCH --gpus=a100:1
#
# Specify disk limit on local scratch.
#SBATCH --tmp=500000

echo "=== Job starting on $(hostname) at $(date) ==="
# DATE_VAR=$(date +%Y%m%d%H%M%S)

# Load modules.
module load stack/2024-06 python/3.12 cuda/12.4 eth_proxy
echo "Loaded modules: $(module list 2>&1)"

# Activate virtual environment for sam2.
source /cluster/scratch/niacobone/map-anything/myenv/bin/activate
echo "Activated Python venv: $(which python)"

# check if the dataset is available in /cluster/scratch/niacobone/distillation/coco2017
if [ ! -d "/cluster/scratch/niacobone/distillation/coco2017" ]; then
    echo "Dataset not found in /cluster/scratch/niacobone/distillation/coco2017 - copyting from /cluster/work/igp_psr/niacobone/coco2017"
    cp -r /cluster/work/igp_psr/niacobone/coco2017 /cluster/scratch/niacobone/distillation/
    echo "Dataset copied."
fi

# Execute
cd /cluster/scratch/niacobone/map-anything
echo "Starting MapAnything distillation..."

export WANDB_API_KEY=$(cat "/cluster/home/niacobone/.config/wandb/wandb_api_key.txt")

python -u distillation.py --wandb_name "run_5_distillation_branch25E" --load_checkpoint "checkpoint_epoch25.pth" --lr 1e-5 --branch_wandb_run "re3zfxdy"

echo "=== Job finished at $(date) ==="
start_time=${SLURM_JOB_START_TIME:-$(date +%s)}
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) (hh:mm:ss)"