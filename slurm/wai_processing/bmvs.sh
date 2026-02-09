#!/bin/bash
#
#SBATCH --job-name=wai_bmvs_processing
#SBATCH --output=wai_bmvs_processing_%j.log
#SBATCH --error=wai_bmvs_processing_%j.err
#SBATCH --open-mode=append
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --gpus=rtx_4090:1
#SBATCH --tmp=500000

# Specify email notifications.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=niacobone@student.ethz.ch

echo "=== Job starting on $(hostname) at $(date) ==="

# Carica i moduli necessari
module load stack/2024-06 python/3.12 cuda/12.4.1 eth_proxy

# Attiva il venv
source /cluster/scratch/niacobone/map-anything/wai_processing/bin/activate
echo "Activated Python venv: $(which python)"

cd /cluster/scratch/niacobone/map-anything

# # 1. Conversione blendedmvs
# python -m wai_processing.scripts.conversion.blendedmvs \
#  original_root="/cluster/scratch/niacobone/distillation/dataset/blendedmvs" \
#  root="/cluster/scratch/niacobone/distillation/dataset/processed/blendedmvs"

# Copia covisibility da blendedmvs_integration a blendedmvs
# for scene in /cluster/scratch/niacobone/distillation/dataset/processed/blendedmvs_integration/*; do
#   scene_name=$(basename "$scene")
#   if [ -d "$scene/covisibility" ]; then
#     mkdir -p /cluster/scratch/niacobone/distillation/dataset/processed/blendedmvs/"$scene_name"
#     cp -r "$scene/covisibility" /cluster/scratch/niacobone/distillation/dataset/processed/blendedmvs/"$scene_name"/
#   fi
# done

# # 2. Covisibility
# python -m wai_processing.scripts.covisibility \
#   data_processing/wai_processing/configs/covisibility/covisibility_gt_depth_224x224.yaml \
#   root="/cluster/scratch/niacobone/distillation/dataset/processed/blendedmvs"

# 3. MoGe
# python -m wai_processing.scripts.run_moge \
#  root="/cluster/scratch/niacobone/distillation/dataset/processed/blendedmvs" \
#  batch_size=1

scenes=(/cluster/scratch/niacobone/distillation/dataset/processed/blendedmvs/*)
total=${#scenes[@]}
count=0

for d in "${scenes[@]}"; do
  scene=$(basename "$d")
  src="/cluster/scratch/niacobone/distillation/dataset/processed/blendedmvs/$scene"
  dst="/cluster/scratch/niacobone/distillation/dataset/processed/blendedmvs_integration/$scene"
  ((count++))
  echo -ne "[$count/$total] Copio $scene\r"
  # Copia solo se non esiste già
  if [ ! -f "$dst/scene_meta.json" ]; then
    cp "$src/scene_meta.json" "$dst/"
  fi
  # Copia 1:1 la cartella moge (solo file mancanti)
  if [ -d "$src/moge" ]; then
    mkdir -p "$dst/moge"
    rsync -a --ignore-existing "$src/moge/" "$dst/moge/"
  fi
done
echo -e "\nFatto!"

echo "=== Job finished at $(date) ==="
start_time=${SLURM_JOB_START_TIME:-$(date +%s)}
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) (hh:mm:ss)"