import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from skopt.learning import ExtraTreesRegressor
from skopt import Optimizer
from file_paths_and_consts import *
import os
import geopandas as gpd
import pandas as pd
import json
import sys
import argparse
import warnings
import pickle

warnings.filterwarnings('ignore')
def get_return_df(df):
    df['Total'] = df[DEMO_TYPES].sum(axis=1)
    df['return_date'] = pd.to_datetime(df['return_date'])
    return_df = df.groupby('return_date')['Total'].sum().reset_index()
    return_df['Total'] = return_df['Total'].rolling(7).mean()
    return_df = return_df.dropna(subset=['Total'])
    return return_df

## returns the daily return estimation for a specific param value
def get_estimation(param,return_dir_prefix,all_raions,dest_name,Q1=0.2,Q3=0.8,cache_clean_mode=0):
    RETURN_DIR = f'{OUTPUT_DIR}{return_dir_prefix}_{param}/'
    cache_file_name = f'calib_return_est_from_{dest_name}_to_UKR_method_{return_dir_prefix}_from_{param}.pq'
    if os.path.isfile(CACHE_DIR+cache_file_name) and cache_clean_mode==0:
        print('cache found for',RETURN_DIR,flush=True)
        df = pd.read_parquet(CACHE_DIR+cache_file_name)
        df['return_date'] = pd.to_datetime(df['return_date'])
        return df 
    
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
            print('gathered',len(cur_settings_df),'results for',sim_idx,'in',RETURN_DIR,flush=True)
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
    return return_uncertain_df

def compute_loss(sim_df, observed_df, observe_col, date_start, date_end, date_col='return_date',gt_scale=1.0,lcorr=0.005):
    merged_df = sim_df.merge(observed_df,on=date_col,how='inner')
    merged_df = merged_df[(merged_df[date_col]>=pd.to_datetime(date_start)) & (merged_df[date_col]<=pd.to_datetime(date_end))]
    merged_df[observe_col] = merged_df[observe_col]/gt_scale
    mse = (((merged_df['Total']-merged_df[observe_col])**2).sum())/merged_df.shape[0]
    corr = merged_df['Total'].corr(merged_df[observe_col], method='pearson')
    rmse = mse**0.5
    nrmse = rmse/(max(merged_df[observe_col])-min(merged_df[observe_col]))
    #print(nrmse,corr,rmse)
    return lcorr*(1-corr)+0.8*nrmse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_param1", type=float, required=True)
parser.add_argument("--input_param2", type=float, required=True)
parser.add_argument("--input_param3", type=float, required=True)
parser.add_argument("--it", type=int, required=True)
parser.add_argument("--dir_prefix", type=str, required=True)
parser.add_argument("--functiontype", type=str, required=True)
args = parser.parse_args()
print('arguments parsed..',flush=True)

## reading ground truth data
geo_shp_file = BASE_DIR+'raw_data/UKR_shapefile_2/ukr_shp/ukr_admbnda_adm2_sspe_20230201.shp'
ukr_gdf = gpd.read_file(geo_shp_file)
all_raions = ukr_gdf['ADM2_EN'].tolist()
print('regions read..',flush=True)
## poland gt data
pl_border_data = pd.read_csv('poland_border_movement_utf8.csv',thousands=',')
#ukr_people_arrive_poland_by_date = ukr_people_arrive_poland_by_date.dropna(subset=['Total'])
pl_border_data['Date'] = pd.to_datetime(pl_border_data['Date'])
ukr_people_arrive_poland_by_date = ((pl_border_data[(pl_border_data.Direction=='Arrival to Poland') & (pl_border_data['Citizenship (Code)']=='UA')]).groupby('Date')['Total'].sum()).reset_index()
ukr_people_arrive_poland_by_date = ukr_people_arrive_poland_by_date.sort_values(by='Date')
ukr_people_depart_poland_by_date = ((pl_border_data[(pl_border_data.Direction=='Departure from Poland') & (pl_border_data['Citizenship (Code)']=='UA')]).groupby('Date')['Total'].sum()).reset_index()
ukr_people_depart_poland_by_date = ukr_people_depart_poland_by_date.sort_values(by='Date')
ukr_people_arrive_poland_by_date['Total'] = ukr_people_arrive_poland_by_date['Total'].rolling(15).mean()
ukr_people_depart_poland_by_date['Total'] = ukr_people_depart_poland_by_date['Total'].rolling(15).mean()
ukr_people_arrive_poland_by_date = ukr_people_arrive_poland_by_date.dropna(subset='Total')
ukr_people_depart_poland_by_date = ukr_people_depart_poland_by_date.dropna(subset='Total')
ukr_people_arrive_poland_by_date =  ukr_people_arrive_poland_by_date.rename(columns={'Date':'return_date','Total':'arrival'})
ukr_people_depart_poland_by_date =  ukr_people_depart_poland_by_date.rename(columns={'Date':'return_date','Total':'departure'})

print('initializing optimizer..',flush=True)

optimizer_name = f'return-optimizer-{args.dir_prefix}-{args.functiontype}.pkl'
dir_name = f'RETURN_HAZARD_{args.dir_prefix}_{args.functiontype}'
out_file_name = f'next_param_suggestion_{args.dir_prefix}_{args.functiontype}_{args.it}.opt'

if os.path.isfile(optimizer_name):
    with open(optimizer_name, 'rb') as f:
        opt = pickle.load(f)
else:
    opt = Optimizer([(0, 0.0008),(0.0,0.008),(0.1,1.0)], "GP", acq_func="EI",acq_optimizer="sampling",random_state=42)

print('old param',args.input_param1,args.input_param2,args.input_param3,flush=True)

try:
    sim_df = get_estimation(str(args.it).zfill(9),dir_name,all_raions,'PL',cache_clean_mode=1)
except:
    next_x = [args.input_param1,args.input_param2,args.input_param3]   
    with open(out_file_name,'w') as opt_f:
        for p in next_x:
            print(f"{p:.8f}",file=opt_f)
    opt_f.close()
    exit(0)
ll = compute_loss(sim_df, ukr_people_depart_poland_by_date, 'departure', '2022-04-01', '2022-08-01')
print('loss computed..',flush=True)
opt.tell([args.input_param1,args.input_param2,args.input_param3],ll)

while True:
    rr = opt.ask()
    if rr in opt.get_result()['x_iters']:
        print(rr,'already evaluated',flush=True)
        continue
    else:
        break

#next_x = rr[0]
with open(out_file_name,'w') as opt_f:
    for p in rr:
        print(f"{p:.8f}",file=opt_f)
opt_f.close()

with open(optimizer_name, 'wb') as f:
    pickle.dump(opt, f)
