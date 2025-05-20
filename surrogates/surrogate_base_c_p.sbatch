#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH -t 12:00:00
#SBATCH -p bii
#SBATCH -A nssac_covid19
#SBATCH --job-name=sbcp

module load miniforge

conda activate migration_env

python surrogate_base_c_p.py  $1 $2 $3 $4
