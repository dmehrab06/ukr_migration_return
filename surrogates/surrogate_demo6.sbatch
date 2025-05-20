#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH -t 12:00:00
#SBATCH -p bii
#SBATCH -A nssac_covid19
#SBATCH --job-name=sg_demo6

module load miniforge

conda activate migration_env

python surrogate_demographic_4.py  $1
