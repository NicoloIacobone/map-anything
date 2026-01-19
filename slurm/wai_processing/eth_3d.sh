#!/bin/bash
#
#SBATCH --job-name=wai_eth3d_processing
#SBATCH --output=wai_eth3d_processing_%j.log
#SBATCH --error=wai_eth3d_processing_%j.err
#SBATCH --open-mode=append
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --gpus=rtx_4090:1
#SBATCH --tmp=500000

echo "=== Job starting on $(hostname) at $(date) ==="

# Carica i moduli necessari
module load stack/2024-06 python/3.12 cuda/12.4.1 eth_proxy

# Attiva il venv
source /cluster/scratch/niacobone/map-anything/wai_processing/bin/activate
echo "Activated Python venv: $(which python)"

cd /cluster/scratch/niacobone/map-anything

# # 1. Conversione ETH3D
# python -m wai_processing.scripts.conversion.eth3d \
#   original_root="/cluster/scratch/niacobone/distillation/dataset/normal/ETH3D" \
#   root="/cluster/scratch/niacobone/distillation/dataset/processed/ETH3D"

# # 2. Covisibility
# python -m wai_processing.scripts.covisibility \
#   data_processing/wai_processing/configs/covisibility/covisibility_gt_depth_224x224.yaml \
#   root="/cluster/scratch/niacobone/distillation/dataset/processed/ETH3D"

# 3. MoGe
python -m wai_processing.scripts.run_moge \
  root="/cluster/scratch/niacobone/distillation/dataset/processed/ETH3D" \
  batch_size=1

echo "=== Job finished at $(date) ==="
start_time=${SLURM_JOB_START_TIME:-$(date +%s)}
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) (hh:mm:ss)"