!/usr/bin/bash
#SBATCH -n 1
#SBATCH --time=01:00:00 
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=100g 

# Copy files to local scratch
rsync -aq ./ ${TMPDIR}
# Run commands
cd $TMPDIR
# Command to run the job that processes the data
do_my_calculation
# Copy new and changed files back.
# Slurm saves the path of the directory from which the job was submitted in $SLURM_SUBMIT_DIR
rsync -auq ${TMPDIR}/ $SLURM_SUBMIT_DIR