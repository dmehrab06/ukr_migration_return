#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH -t 02:00:00
#SBATCH -p bii
#SBATCH -A nssac_covid19
#SBATCH --job-name=cache_refresh

module load miniforge
source activate migration_env

#python abm_dest_return.py $1 $2 $3 $4 $5
python cache_refresh_for_dest_return.py $1 $2 $3

