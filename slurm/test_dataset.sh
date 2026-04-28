#!/bin/bash
#
# Specify job name.
#SBATCH --job-name=distillation_blendedmvs
#
# Specify output file.
#SBATCH --output=distillation_blendedmvs_%j.log
#
# Specify error file.
#SBATCH --error=distillation_blendedmvs_%j.err
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
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=niacobone@student.ethz.ch

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
 
# Copia dataset nella cartella temporanea del job, lo decomprime e conta cartelle e immagini
DATASET_SRC="/cluster/work/igp_psr/niacobone/distillation/dataset/backup/blendedmvs_dataset.tar.gz"
LOCAL_TAR="$TMPDIR/$(basename "$DATASET_SRC")"

# Copia
echo "Copying dataset from $DATASET_SRC to $LOCAL_TAR"
START_COPY=$(date +%s)
cp "$DATASET_SRC" "$LOCAL_TAR"
END_COPY=$(date +%s)
COPY_TIME=$((END_COPY - START_COPY))
echo "Tempo impiegato per la copia: ${COPY_TIME}s"
if [ ! -f "$LOCAL_TAR" ]; then
	echo "Errore: copia del file fallita: $LOCAL_TAR" >&2
	exit 1
fi

# Determina directory principale nell'archivio
echo "Listing archive top-level entries to determine extraction path"
START_LIST=$(date +%s)
TOP_DIR=$(tar -tzf "$LOCAL_TAR" | sed -e 's@/.*@@' | uniq | head -n1)
END_LIST=$(date +%s)
LIST_TIME=$((END_LIST - START_LIST))
echo "Tempo impiegato per determinare la directory principale: ${LIST_TIME}s"
if [ -z "$TOP_DIR" ]; then
	echo "Errore: impossibile determinare la directory principale nell'archivio" >&2
	exit 1
fi

# Estrazione
echo "Extracting $LOCAL_TAR to $TMPDIR"
START_EXTRACT=$(date +%s)
tar -xzf "$LOCAL_TAR" -C "$TMPDIR"
END_EXTRACT=$(date +%s)
EXTRACT_TIME=$((END_EXTRACT - START_EXTRACT))
echo "Tempo impiegato per l'estrazione: ${EXTRACT_TIME}s"
EXTRACT_PATH="$TMPDIR/$TOP_DIR"

if [ ! -d "$EXTRACT_PATH" ]; then
	echo "Errore: directory estratta non trovata: $EXTRACT_PATH" >&2
	exit 1
fi

# Conteggio
echo "Conteggio delle cartelle e delle immagini in: $EXTRACT_PATH"
START_COUNT=$(date +%s)
TOTAL_DIRS=$(find "$EXTRACT_PATH" -type d | wc -l)
TOTAL_IMAGES=$(find "$EXTRACT_PATH" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) | wc -l)
END_COUNT=$(date +%s)
COUNT_TIME=$((END_COUNT - START_COUNT))
echo "Tempo impiegato per il conteggio: ${COUNT_TIME}s"

echo "Numero totale di cartelle: $TOTAL_DIRS"
echo "Numero totale di immagini: $TOTAL_IMAGES"

exit 0

