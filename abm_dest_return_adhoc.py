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
SEED = (int(sys.argv[5]) if len(sys.argv)>=6 else 0)
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = '/project/biocomplexity/UKR_forecast/migration_data/'
geo_shp_file = BASE_DIR+'raw_data/UKR_shapefile_2/ukr_shp/ukr_admbnda_adm2_sspe_20230201.shp'
ukr_gdf = gpd.read_file(geo_shp_file)

raion_to_dest_df = pd.read_csv('from_raion_to_dest_refugee_pdf.csv')
default_raion = raion_to_dest_df['ADM2_EN'].tolist()[0]
#print(default_raion)

USE_NEIGHBOR = 5
CONFLICT_DATA_PREFIX = 'ukraine_conflict_data_ADM2_HDX_buffer_'
NEIGHBOR_DATA_PREFIX = 'ukraine_neighbor_'
NETWORK_TYPE = '_R_0.01_P_0.04_Q_8_al_2.3.csv'

sim_idx = str(11).zfill(9)

total_impact_data = pd.read_csv(IMPACT_DIR+CONFLICT_DATA_PREFIX+str(USE_NEIGHBOR)+'_km.csv')

def sigmoid(q,v,x):
    if x<=0:
        return 0
    return 1.0 / (1 + q*math.exp(-v*x))

def get_return_prob_demographic(member_info,family_movement,non_family_movement): ##this function can be played with
    tot_size = sum(member_info)
    move_prob_family = sum([x*(1-y) for x,y in zip(member_info,family_movement)])/tot_size
    move_prob_single = sum([x*(1-y) for x,y in zip(member_info,non_family_movement)])/tot_size
    return (move_prob_family if tot_size>1 else move_prob_single)

def get_return_prob(hid,cur_time,move_date,normalized_conflict_count,member_info):
    days_passed = (cur_time-move_date).days
    time_passed_weight = sigmoid(Q_R,V_R,days_passed-1)
    conflict_observation_weight = (1-normalized_conflict_count)*get_return_prob_demographic(member_info,FAMILY_PROB,MOVE_PROB)
    #if hid==3626417:
    #    print('time passed weight',time_passed_weight)
    #    print('conflict observe weight',conflict_observation_weight)
    #    print('tot_prob',SCALE_WEIGHT*time_passed_weight*conflict_observation_weight)
    return SCALE_WEIGHT*time_passed_weight*conflict_observation_weight

def get_coin_side(prob_of_head,p):
    return 1 if p<=prob_of_head else 0


def get_refugee_file(raion_name,sim_idx):
    refugee_file_name = f'mim_hid_completed_{raion_name}_{sim_idx}.csv'
    hid_status_file = OUTPUT_DIR+refugee_file_name
    df_hid = pd.read_csv(hid_status_file)
    df_hid = df_hid[['hid','prob_conflict','OLD_PERSON','CHILD','ADULT_MALE','ADULT_FEMALE','rlid','h_lat','h_lng',
                     'N_size','P(move|violence)','moves','move_type','move_date']]
    df_refugee = df_hid[df_hid.move_type==2]
    #print(df_hid.shape[0],'displaced from MIM simulation')
    #print(df_refugee.shape[0],'refugees from MIM simulation')
    return df_refugee

def assign_destination(df_refugee,raion_to_dest_df,raion_name):
    sci_country_code = ['HU','MDA','PL','RO','SK','BLR']
    dest_factors = ['weight_sci','weight_nato','weight_gdp','weight_gdp_per_capita','weight_dis']
    country_3_code = ['HUN','MDA','POL','ROU','SVK','BLR']
    
    if raion_name in raion_to_dest_df['ADM2_EN'].tolist():
        print(raion_name,'contains a prob distribution')
        dest_pdf = raion_to_dest_df[raion_to_dest_df.ADM2_EN==raion_name]
        #default_pdf['normalized_weight'] = 1.0/len(sci_country_code)
    else:
        print(raion_name,'does not exist, loading uniform distribution')
        dest_pdf = raion_to_dest_df[raion_to_dest_df.ADM2_EN==default_raion]
        dest_pdf['normalized_weight'] = 1.0/len(sci_country_code)
        
    dest_list = (dest_pdf.sample(df_refugee.shape[0],replace=True,weights=dest_pdf['normalized_weight'])['dest']).tolist()
    df_refugee['dest'] = dest_list
    return df_refugee



MOVE_PROB = [0.25,0.7,0.02,0.7]
FAMILY_PROB = [0.25,0.85,0.1,0.85]
SCALE_WEIGHT = float(sys.argv[2])
Q_R = float(sys.argv[3])
V_R = float(sys.argv[4])

RETURN_DIR_NAME = f'RETURN_PARAM_WITH_SEED_WITH_PEER_{SCALE_WEIGHT}_{Q_R}_{V_R}/'

if not os.path.isdir(OUTPUT_DIR+RETURN_DIR_NAME):
    os.makedirs(OUTPUT_DIR+RETURN_DIR_NAME)

PARENT_REGION = str(sys.argv[1])
with open('raion_partitions_10.json') as f_in:
    REGION_BIN_DICT = json.load(f_in)
        
ALL_RAION_BINS = [PARENT_REGION]+REGION_BIN_DICT[PARENT_REGION]
print('will process the raions:',ALL_RAION_BINS,flush=True)

##more optimized code
## return_prob is calculated without the function
for raion_name in ALL_RAION_BINS:
    try:
        returned_refugees = []
        st_time = time.time()

        if os.path.isfile(HOUSEHOLD_DIR+'KSW_HH_BALL_AAMAS_'+raion_name+NETWORK_TYPE):
            print('KSW neighborhood loaded')
            neighbor_household_data = pd.read_csv(HOUSEHOLD_DIR+'KSW_HH_BALL_AAMAS_'+raion_name+NETWORK_TYPE)
        else:
            print('s2 neighborhood loaded')
            neighbor_household_data = pd.read_csv(HOUSEHOLD_DIR+NEIGHBOR_DATA_PREFIX+raion_name+'_13_s2.csv',usecols=['hid_x','hid_y'])

        df_refugee = get_refugee_file(raion_name,sim_idx)
        df_refugee = assign_destination(df_refugee,raion_to_dest_df,raion_name)
        df_refugee['origin'] = raion_name

        START_DATE = '2022-02-24'
        END_DATE = '2022-09-01'
        T_CURRENT = pd.to_datetime(START_DATE)
        T_FINAL = pd.to_datetime(END_DATE)
        NO_RETURN_DATE = pd.to_datetime('2025-01-01')

        impact_data = total_impact_data[total_impact_data.matching_place_id==raion_name]
        impact_data['time'] = pd.to_datetime(impact_data['time'])
        df_refugee['move_date'] = pd.to_datetime(df_refugee['move_date'])
        df_refugee['return_date'] = pd.to_datetime('2025-01-01')
        df_refugee['demo_move_prob'] = df_refugee.apply(lambda x: get_return_prob_demographic(x[DEMO_TYPES].tolist(),FAMILY_PROB,MOVE_PROB),axis=1)

        hid_nsize = (neighbor_household_data.groupby('hid_x')['hid_y'].count().reset_index()).rename(columns={'hid_x':'hid','hid_y':'n_size'})
        refugee_nsizes_total = hid_nsize.merge(df_refugee[['hid']],on='hid',how='inner')
        refugee_neighborhood_only = neighbor_household_data.merge(df_refugee[['hid']],left_on='hid_x',right_on='hid',how='inner')
        refugee_neighborhood_only = refugee_neighborhood_only.drop(columns=['hid'])
        refugee_neighborhood_only = refugee_neighborhood_only.merge(df_refugee[['hid']],left_on='hid_y',right_on='hid',how='inner')
        refugee_neighborhood_only = refugee_neighborhood_only.drop(columns=['hid'])
        refugee_nsizes_outside = (refugee_neighborhood_only.groupby('hid_x')['hid_y'].count().reset_index()).rename(columns={'hid_x':'hid','hid_y':'n_size_migrant'})
        neighbor_effect_df = refugee_nsizes_outside.merge(refugee_nsizes_total,on='hid',how='inner')
        neighbor_effect_df['peer_return_fraction'] = (neighbor_effect_df['n_size']-neighbor_effect_df['n_size_migrant'])/neighbor_effect_df['n_size']

        print('simulation starting...')

        for i in range(0,300):
            #print(T_CURRENT)
            if T_CURRENT > T_FINAL:
                break
            sim_start = time.time()
            cur_impact_count = impact_data[impact_data.time==T_CURRENT].shape[0]
            tot_count = impact_data.shape[0]
            normalized_impact_count = cur_impact_count/(tot_count+1)

            ret_module_start = time.time()
            #approach 1: takes 80 seconds
            #df_refugee['return_prob'] = df_refugee.apply(lambda x: get_return_prob(x['hid'],T_CURRENT,x['move_date'],normalized_impact_count,x[DEMO_TYPES].tolist()),axis=1)

            #approach 2: takes -- seconds
            df_refugee['days_passed'] = (T_CURRENT-df_refugee['move_date']).dt.days
            df_refugee['sigmoid_days_passed'] = 1.0/(1+Q_R*np.exp(-V_R*(df_refugee['days_passed']-1)))
            df_refugee['valid_day'] = df_refugee['days_passed'].apply(lambda x: 1 if x>=1 else 0)
            df_refugee['time_weight'] = df_refugee['valid_day']*df_refugee['sigmoid_days_passed']
            df_refugee = df_refugee.merge(neighbor_effect_df[['hid','peer_return_fraction']],on='hid',how='inner')
            df_refugee['return_prob'] = SCALE_WEIGHT*df_refugee['time_weight']*(1-normalized_impact_count)*df_refugee['demo_move_prob']*df_refugee['peer_return_fraction']
            #df_refugee['return_prob'] = SCALE_WEIGHT*df_refugee['time_weight']*(1-normalized_impact_count)*df_refugee['demo_move_prob']
            df_refugee = df_refugee.drop(columns=['days_passed','sigmoid_days_passed','valid_day','time_weight'])

            ret_module_end = time.time()
            #print(T_CURRENT,'return module calculated in',ret_module_end-ret_module_start,'seconds with pop size',df_refugee.shape[0])

            coin_toss_start = time.time()
            df_refugee['coin_toss'] = np.random.random(df_refugee.shape[0])
            #print(T_CURRENT,'coin toss done in',time.time()-coin_toss_start,'seconds with pop size',df_refugee.shape[0])

            coin_sample_start = time.time()
            df_refugee['return'] = df_refugee.apply(lambda x: get_coin_side(x['return_prob'],x['coin_toss']),axis=1)
            #print(T_CURRENT,'coin side sampled in',time.time()-coin_sample_start,'seconds with pop size',df_refugee.shape[0])

            date_assigned_start = time.time()
            df_refugee['return_date'] = df_refugee['return'].apply(lambda x: T_CURRENT if x==1 else NO_RETURN_DATE)
            #print(T_CURRENT,'return date assigned in',time.time()-date_assigned_start,'seconds with pop size',df_refugee.shape[0])

            split_start = time.time()
            returned_refugees.append(df_refugee[df_refugee['return']==1])
            print(T_CURRENT,df_refugee[df_refugee['return']==1].shape[0],'refugee returned','time taken:',time.time()-sim_start,flush=True)
            df_refugee = df_refugee[df_refugee['return']==0]
            #print(T_CURRENT,'dataset split in',time.time()-split_start,'seconds with remaining pop size',df_refugee.shape[0])
            #print(df_refugee[['hid','return_prob']])

            refugee_neighborhood_only = refugee_neighborhood_only.merge(df_refugee[['hid']],left_on='hid_x',right_on='hid',how='inner')
            refugee_neighborhood_only = refugee_neighborhood_only.drop(columns=['hid'])
            refugee_neighborhood_only = refugee_neighborhood_only.merge(df_refugee[['hid']],left_on='hid_y',right_on='hid',how='inner')
            refugee_neighborhood_only = refugee_neighborhood_only.drop(columns=['hid'])
            refugee_nsizes_outside = (refugee_neighborhood_only.groupby('hid_x')['hid_y'].count().reset_index()).rename(columns={'hid_x':'hid','hid_y':'n_size_migrant'})
            neighbor_effect_df = neighbor_effect_df.drop(columns=['n_size_migrant'])
            neighbor_effect_df = refugee_nsizes_outside.merge(neighbor_effect_df,on='hid',how='inner')
            neighbor_effect_df['peer_return_fraction'] = (neighbor_effect_df['n_size']-neighbor_effect_df['n_size_migrant'])/neighbor_effect_df['n_size']
            df_refugee =  df_refugee.drop(columns=['peer_return_fraction'])

            T_CURRENT = T_CURRENT + pd.DateOffset(days=1)

        print(raion_name,'simulated in',time.time()-st_time,'seconds',flush=True)
        pd.concat(returned_refugees).to_csv(OUTPUT_DIR+RETURN_DIR_NAME+f'mim_hid_return_{raion_name}_SIM_{sim_idx}_SEED_{SEED}.csv',index=False)
    except Exception as e:
        print(e)
        continue
    
