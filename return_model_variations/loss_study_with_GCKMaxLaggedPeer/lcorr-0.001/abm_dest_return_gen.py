import pandas as pd
from file_paths_and_consts import *
import os
import numpy as np
import geopandas as gpd
import math
import warnings
import time
import sys
import random
import json
warnings.filterwarnings('ignore')
#OUTPUT_DIR
## for reporducibility

calibration_iteration = int(sys.argv[1])
H_rate = float(sys.argv[2])
function_type = str(sys.argv[3])
job_name = str(sys.argv[4])
dir_prefix = str(sys.argv[5])
good_sims = [5,6,7,9,11,12,14,19,21,23,26,29,39,40,43,45,46,48,49,50,51,53,54,63,67,78,86,87,98]

with open('raion_partitions_10.json') as f_in:
    REGION_BIN_DICT = json.load(f_in)
        
if function_type in ['constant','constant_conflict','constant_global_conflict','constant_global_conflict_intensity','constant_conflict_intensity']:
    for r in REGION_BIN_DICT:
        for ss in good_sims:
            print('sbatch -J',job_name,'abm_dest_return.sbatch',calibration_iteration,r,function_type+'#'+dir_prefix,H_rate,0.01,ss)
            
if function_type in ['constant_global_conflict_weighted_intensity','weibull_conflict','constant_discounted_conflict','constant_global_conflict_lagged_intensity']:
    V_rate = float(sys.argv[6])
    for r in REGION_BIN_DICT:
        for ss in good_sims:
            print('sbatch -J',job_name,'abm_dest_return.sbatch',calibration_iteration,r,function_type+'#'+dir_prefix,H_rate,V_rate,ss)
            
if ('peer' in function_type) and ('split' not in function_type):
    PEER_Q_R = float(sys.argv[6])
    PEER_T = float(sys.argv[7])
    for r in REGION_BIN_DICT:
        for ss in good_sims:
            print('sbatch -J',job_name,'abm_dest_return.sbatch',calibration_iteration,r,function_type+'#'+dir_prefix,H_rate,0.01,PEER_Q_R,PEER_T,ss)
            
if function_type in ['constant_global_conflict_intensity_split']:
    V_rate = float(sys.argv[6])
    for r in REGION_BIN_DICT:
        for ss in good_sims:
            print('sbatch -J',job_name,'abm_dest_return.sbatch',calibration_iteration,r,function_type+'#'+dir_prefix,H_rate,V_rate,ss)
            
if 'lagged' in function_type and 'split' in function_type:
    V_rate = float(sys.argv[6])
    for r in REGION_BIN_DICT:
        for ss in good_sims:
            print('sbatch -J',job_name,'abm_dest_return.sbatch',calibration_iteration,r,function_type+'#'+dir_prefix,H_rate,V_rate,ss)
