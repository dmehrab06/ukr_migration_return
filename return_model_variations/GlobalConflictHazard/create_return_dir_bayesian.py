import sys
import os
from file_paths_and_consts import *

calibration_it = int(sys.argv[1])
function_type = str(sys.argv[2])
dir_prefix = str(sys.argv[3])

RETURN_DIR_NAME = f'RETURN_HAZARD_{dir_prefix}_{function_type}_{str(calibration_it).zfill(9)}/'

if not os.path.isdir(OUTPUT_DIR+RETURN_DIR_NAME):
    os.makedirs(OUTPUT_DIR+RETURN_DIR_NAME)
