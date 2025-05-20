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

def get_return_df(df):
    df['Total'] = df[DEMO_TYPES].sum(axis=1)
    df['return_date'] = pd.to_datetime(df['return_date'])
    return_df = df.groupby('return_date')['Total'].sum().reset_index()
    return_df['Total'] = return_df['Total'].rolling(7).mean()
    return_df = return_df.dropna(subset=['Total'])
    return return_df

def get_estimation(param,return_dir_prefix,all_raions,dest_name,Q1=0.2,Q3=0.8,cache_clean_mode=0):
    RETURN_DIR = f'{OUTPUT_DIR}{return_dir_prefix}_{param}/'
    cache_file_name = f'calib_return_est_from_{dest_name}_to_UKR_method_{return_dir_prefix}_from_{param}.pq'
    if os.path.isfile(CACHE_DIR+cache_file_name) and cache_clean_mode==0:
        #print('cache found for',RETURN_DIR,flush=True)
        df = pd.read_parquet(CACHE_DIR+cache_file_name)
        df['return_date'] = pd.to_datetime(df['return_date'])
        return df 
    #else:
    #    return
    print('No file called',CACHE_DIR+cache_file_name,'calculating...')
    #print('return dir is',RETURN_DIR,flush=True)
    single_sim_return_df = []
    for sim in range(0,100):
        sim_idx = str(sim).zfill(9)
        base_file = f'mim_hid_return_Kyiv_SIM_{sim_idx}_SEED_0.csv'
        base_file_fast = f'mim_hid_return_Kyiv_SIM_{sim_idx}_SEED_0.pq'
        #print('base_file_loc',RETURN_DIR+base_file,'khujtesi')
        if os.path.isfile(RETURN_DIR+base_file) or os.path.isfile(RETURN_DIR+base_file_fast):
            cur_settings_df = []
            for raion_name in all_raions:
                fname = f'mim_hid_return_{raion_name}_SIM_{sim_idx}_SEED_0.csv'
                fname_faster = f'mim_hid_return_{raion_name}_SIM_{sim_idx}_SEED_0.pq'
                #print('looking for',RETURN_DIR+fname_faster,flush=True)
                if os.path.isfile(RETURN_DIR+fname_faster):
                    #print(RETURN_DIR+fname_faster,'found',flush=True)
                    df = pd.read_parquet(RETURN_DIR+fname_faster)
                    df = df[df.dest==dest_name]
                    cur_settings_df.append(get_return_df(df))
                elif os.path.isfile(RETURN_DIR+fname):
                    df = pd.read_csv(RETURN_DIR+fname)
                    df.to_parquet(RETURN_DIR+fname_faster,index=False)
                    df = df[df.dest==dest_name]
                    cur_settings_df.append(get_return_df(df))
            #print('gathered',len(cur_settings_df),'results for',sim_idx,'in',RETURN_DIR,flush=True)
            if len(cur_settings_df)>0:
                all_return_df = pd.concat(cur_settings_df)
                all_return_df = all_return_df.groupby('return_date')['Total'].sum().reset_index()
                single_sim_return_df.append(all_return_df)
    return_ensemble_df = pd.concat(single_sim_return_df)
    median_df = return_ensemble_df.groupby('return_date')['Total'].quantile(q=0.5).reset_index()
    q1_df = (return_ensemble_df.groupby('return_date')['Total'].quantile(q=Q1).reset_index()).rename(columns={'Total':'q1'})
    q3_df = (return_ensemble_df.groupby('return_date')['Total'].quantile(q=Q3).reset_index()).rename(columns={'Total':'q3'})
    return_uncertain_df = (median_df.merge(q1_df,on='return_date',how='inner')).merge(q3_df,on='return_date',how='inner')
    return_uncertain_df.to_parquet(CACHE_DIR+cache_file_name,index=False)
    #return return_uncertain_df

## reading ground truth data
geo_shp_file = BASE_DIR+'raw_data/UKR_shapefile_2/ukr_shp/ukr_admbnda_adm2_sspe_20230201.shp'
ukr_gdf = gpd.read_file(geo_shp_file)
all_raions = ukr_gdf['ADM2_EN'].tolist()

param = str(sys.argv[1]).zfill(9)
return_dir_prefix = str(sys.argv[2])
dest_name = str(sys.argv[3])

get_estimation(param,return_dir_prefix,all_raions,dest_name,cache_clean_mode=1)
