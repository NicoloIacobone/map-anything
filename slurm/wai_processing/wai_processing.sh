#!/bin/bash
#
# Specify job name.
#SBATCH --job-name=wai_processing
#
# Specify output file.
#SBATCH --output=wai_processing_%j.log
#
# Specify error file.
#SBATCH --error=wai_processing_%j.err
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

# Crea il venv solo se non esiste già
if [ ! -d "/cluster/scratch/niacobone/map-anything/wai_processing" ]; then
  python3 -m venv /cluster/scratch/niacobone/map-anything/wai_processing
fi

# Attiva il venv
source /cluster/scratch/niacobone/map-anything/wai_processing/bin/activate

# Aggiorna pip, wheel e setuptools
pip install --upgrade pip wheel setuptools

# Installa torch con CUDA 12.4
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Installa la root del progetto senza dipendenze
cd /cluster/scratch/niacobone/map-anything
pip install --no-deps . --no-build-isolation

# Installa wai_processing in modalità editable con tutte le extra dependencies
cd data_processing/wai_processing/
pip install -e .[all] --no-build-isolation

# Vai nella cartella wai_processing
cd /cluster/scratch/niacobone/map-anything/data_processing/wai_processing/

# Clona la repo mvsanywhere solo se non esiste già
if [ ! -d "third_party/mvsanywhere" ]; then
  mkdir -p third_party
  cd third_party
  git clone https://github.com/arknapit/mvsanywhere.git
  cd ..
fi

# Scarica il checkpoint solo se non esiste già
if [ ! -f "third_party/mvsanywhere/checkpoints/mvsanywhere_hero.ckpt" ]; then
  mkdir -p third_party/mvsanywhere/checkpoints
  cd third_party/mvsanywhere/checkpoints
  wget https://storage.googleapis.com/niantic-lon-static/research/mvsanywhere/mvsanywhere_hero.ckpt
  cd ../../../
fi

# Installa wai_processing con il supporto mvsanywhere
pip install -e .[mvsanywhere] --no-build-isolation

pip install plyfile

echo "=== Job finished at $(date) ==="
start_time=${SLURM_JOB_START_TIME:-$(date +%s)}
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) (hh:mm:ss)"