#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000
#SBATCH -t 00:10:00
#SBATCH -p bii
#SBATCH -A nssac_covid19
#SBATCH --job-name=ret_2_param
#SBATCH --output=ret_2_param_%j.out

bash return_pipeline_double_param.sh $1 $2 $3 $4 $5 $6

