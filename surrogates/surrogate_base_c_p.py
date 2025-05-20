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
import properscoring as ps
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from skopt import Optimizer
import sys
from surrogate_utils import *

warnings.filterwarnings('ignore')
    
geo_shp_file = BASE_DIR+'raw_data/UKR_shapefile_2/ukr_shp/ukr_admbnda_adm2_sspe_20230201.shp'
ukr_gdf = gpd.read_file(geo_shp_file)
all_raions = ukr_gdf['ADM2_EN'].tolist()
print('regions read..',flush=True)

raion_to_dest_df = pd.read_csv('from_raion_to_dest_refugee_pdf.csv')

## reading ground truth data

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

#intention_sim_demo6 = load_intention_destination_df_by_demo_category6(raion_to_dest_df,all_raions)
intention_sim = load_intention_destination_df(raion_to_dest_df,all_raions)

roll = 7
lag = int(sys.argv[4])

conflict_context = get_conflict_data(roll,lag)
random_state = int(sys.argv[1])
lerror = float(sys.argv[2])
lcorr = float(sys.argv[3])

opt = Optimizer([(0.0, 0.005),(0.0, 0.0005),(0.0,1.0)], "GP", acq_func="EI",acq_optimizer="sampling",random_state=42)

optimizer_name = f'return-optimizer-bayes-surrrogate-gc-peer-{lag}-7-{random_state}-{lerror}-{lcorr}'


with open(f'../optimizers/{optimizer_name}', 'rb') as f:
    opt = pickle.load(f)

tot_it = 200
for i in range(0,tot_it):
    rr = opt.ask()
    #print(rr)
    if (i%50==0):
        print('calibration step',i)
    ll,_,_,_,_ = estimate_blackbox_function_loss(intention_sim,ukr_people_depart_poland_by_date,surrogate_conflict_haz_with_peer,
                                                 rr,lerr=lerror,lcorr=lcorr,no_print=1,conflict_context=conflict_context)
    #print(next_h,ll)
    opt.tell(rr,ll)

#print('best result has a loss of',opt.get_result()['fun'],'for',opt.get_result()['x'])

xs = opt.get_result()['x_iters']
ys = opt.get_result()['func_vals']
TOPK = 3
min_observe = findminobservation_idx(xs,ys,k=TOPK)
for idx, xx in enumerate(min_observe):

    _,nrmse,corr,_,_ = estimate_blackbox_function_loss(intention_sim,ukr_people_depart_poland_by_date,
                                                       surrogate_conflict_haz_with_peer,
                                                       xs[xx],lerr=lerror,lcorr=lcorr,
                                                       no_print=1,conflict_context=conflict_context)
    print('model',idx)
    print('random seed:',random_state)
    print('roll:',roll)
    print('lag:',lag)
    print('lerror:',lerror)
    print('lcorr:',lcorr)
    print('nrmse:',nrmse)
    print('corr:',corr)

with open(f'../optimizers/{optimizer_name}', 'wb') as f:
    pickle.dump(opt, f)

