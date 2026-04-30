#!/bin/bash
#
# Specify job name.
#SBATCH --job-name=distillation_full_dataset_3
#
# Specify output file.
#SBATCH --output=distillation_full_dataset_3_%j.log
#
# Specify error file.
#SBATCH --error=distillation_full_dataset_3_%j.err
#
# Specify open mode for log files.
#SBATCH --open-mode=append
#
# Specify time limit.
#SBATCH --time=24:00:00
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
#

echo "=== Job starting on $(hostname) at $(date) ==="
# DATE_VAR=$(date +%Y%m%d%H%M%S)

DATASET_SRC=/cluster/scratch/niacobone/distillation/dataset/backup/blendedmvs_dataset.tar.gz

# Extract dataset to local scratch
echo "Extracting $DATASET_SRC to $TMPDIR"
START_EXTRACT=$(date +%s)
tar -xzf "$DATASET_SRC" -C "$TMPDIR"
END_EXTRACT=$(date +%s)
EXTRACT_TIME=$((END_EXTRACT - START_EXTRACT))
echo "Tempo impiegato per l'estrazione: ${EXTRACT_TIME}s"

# Validazione estrazione
if [ ! -d "$TMPDIR/blendedmvs" ]; then
echo "Errore: directory estratta non trovata: $TMPDIR/blendedmvs" >&2
exit 1
fi

echo "=== Dataset copied and extracted successfully ==="

# Load modules.
module load stack/2024-06 python/3.12 cuda/12.4 eth_proxy
echo "Loaded modules: $(module list 2>&1)"

# Activate virtual environment for sam2.
source /cluster/scratch/niacobone/map-anything/myenv/bin/activate
echo "Activated Python venv: $(which python)"

# Execute
cd /cluster/scratch/niacobone/map-anything/scripts

echo "Starting MapAnything distillation..."

export WANDB_API_KEY=$(cat "/cluster/home/niacobone/.config/wandb/wandb_api_key.txt")

python distill.py \
  machine=cluster \
  machine.base_dir="$TMPDIR" \
  machine.root_data_dir="$TMPDIR/converted/wai_data" \
  machine.mapanything_dataset_metadata_dir="$TMPDIR/converted/mapanything_dataset_metadata" \
  loss=distillation_only \
  train_params.use_wandb=true \
  train_params.run_name=distillation_full_dataset_3

echo "=== Job finished at $(date) ==="
start_time=${SLURM_JOB_START_TIME:-$(date +%s)}
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) (hh:mm:ss)"