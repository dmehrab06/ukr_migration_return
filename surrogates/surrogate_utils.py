from file_paths_and_consts import *
import pandas as pd
import os
import matplotlib.pyplot as plt
import heapq
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import properscoring as ps
from skopt import Optimizer
import sys
import heapq

def findminobservation_idx(xs,ys,k=1):
    return [idx for idx, _ in heapq.nsmallest(k, enumerate(ys), key=lambda x: x[1])]

def get_return_df(df):
    df['Total'] = df[DEMO_TYPES].sum(axis=1)
    df['return_date'] = pd.to_datetime(df['return_date'])
    return_df = df.groupby('return_date')['Total'].sum().reset_index()
    return_df['Total'] = return_df['Total'].rolling(7).mean()
    return_df = return_df.dropna(subset=['Total'])
    return return_df

def assign_destination(df_refugee,raion_to_dest_df,raion_name):
    sci_country_code = ['HU','MDA','PL','RO','SK','BLR']
    dest_factors = ['weight_sci','weight_nato','weight_gdp','weight_gdp_per_capita','weight_dis']
    country_3_code = ['HUN','MDA','POL','ROU','SVK','BLR']
    default_raion = raion_to_dest_df['ADM2_EN'].tolist()[0]
    if raion_name in raion_to_dest_df['ADM2_EN'].tolist():
        #print(raion_name,'contains a prob distribution')
        dest_pdf = raion_to_dest_df[raion_to_dest_df.ADM2_EN==raion_name]
        #default_pdf['normalized_weight'] = 1.0/len(sci_country_code)
    else:
        #print(raion_name,'does not exist, loading uniform distribution')
        dest_pdf = raion_to_dest_df[raion_to_dest_df.ADM2_EN==default_raion]
        dest_pdf['normalized_weight'] = 1.0/len(sci_country_code)
        
    dest_list = (dest_pdf.sample(df_refugee.shape[0],random_state=42,replace=True,weights=dest_pdf['normalized_weight'])['dest']).tolist()
    df_refugee['dest'] = dest_list
    return df_refugee

def get_refugee_by_dest_agg(df_refugee,raion_name,dest='PL'):
    df_refugee['h_size'] = df_refugee[DEMO_TYPES].sum(axis=1)
    df_refugee['move_date'] = pd.to_datetime(df_refugee['move_date'])
    aggdf_refugee_dest = df_refugee[df_refugee.dest=='PL'].groupby('move_date')['h_size'].sum().reset_index()
    return aggdf_refugee_dest

def insiderange(x,val1,val2):
    return (x>=val1 and x<=val2)

def apply_footprint(h_row,footprint):
    val1 = insiderange(h_row['OLD_PERSON'],footprint[0][0],footprint[0][1])
    val2 = insiderange(h_row['CHILD'],footprint[1][0],footprint[1][1])
    val3 = insiderange(h_row['ADULT_MALE'],footprint[2][0],footprint[2][1])
    val4 = insiderange(h_row['ADULT_FEMALE'],footprint[3][0],footprint[3][1])
    return val1 and val2 and val3 and val4

def get_refugee_by_household_category_agg(df_refugee,raion_name,dest='PL'):
    footprints = [[(0,0),(0,0),(1,1),(0,0)],[(0,0),(0,0),(0,0),(1,1)],
                  [(0,0),(0,0),(2,100),(0,0)],[(0,0),(0,0),(0,0),(2,100)],
                  [(1,100),(0,0),(0,100),(0,100)],[(0,0),(1,100),(0,100),(0,100)],
                  [(1,100),(1,100),(0,100),(0,100)],[(0,0),(0,0),(1,100),(1,100)]]
    groups = ['single_male','single_female','male_only','female_only',
              'elderly_no_child','child_no_elderly','child_and_elderly','just_couples']
    
    for idx,g in enumerate(groups):
        df_refugee[g] = df_refugee.apply(lambda x: apply_footprint(x,footprints[idx]),axis=1)
    #return df_refugee
    df_refugee['h_size'] = df_refugee[DEMO_TYPES].sum(axis=1)
    df_refugee['move_date'] = pd.to_datetime(df_refugee['move_date'])
    
    agg_df = (df_refugee[(df_refugee.dest==dest)].groupby('move_date')['h_size'].sum().reset_index())
    
    for idx,g in enumerate(groups):
        cur_demo_agg_df = (df_refugee[(df_refugee.dest==dest) & 
                                      (df_refugee[g]==True)].groupby('move_date')['h_size'].sum().reset_index()).rename(columns={'h_size':g})
        agg_df = agg_df.merge(cur_demo_agg_df,on='move_date',how='left').fillna(0)
    
    return agg_df

def get_refugee_by_household_category6_agg(df_refugee,raion_name,dest='PL'):
    footprints = [[(0,0),(0,0),(1,100),(0,0)],[(0,0),(0,0),(0,0),(1,100)],
                  [(1,100),(0,0),(0,100),(0,100)],[(0,0),(1,100),(0,100),(0,100)],
                  [(1,100),(1,100),(0,100),(0,100)],[(0,0),(0,0),(1,100),(1,100)]]
    groups = ['male_only','female_only',
              'elderly_no_child','child_no_elderly',
              'child_and_elderly','just_couples']
    
    for idx,g in enumerate(groups):
        df_refugee[g] = df_refugee.apply(lambda x: apply_footprint(x,footprints[idx]),axis=1)
    #return df_refugee
    df_refugee['h_size'] = df_refugee[DEMO_TYPES].sum(axis=1)
    df_refugee['move_date'] = pd.to_datetime(df_refugee['move_date'])
    
    agg_df = (df_refugee[(df_refugee.dest==dest)].groupby('move_date')['h_size'].sum().reset_index())
    
    for idx,g in enumerate(groups):
        cur_demo_agg_df = (df_refugee[(df_refugee.dest==dest) & 
                                      (df_refugee[g]==True)].groupby('move_date')['h_size'].sum().reset_index()).rename(columns={'h_size':g})
        agg_df = agg_df.merge(cur_demo_agg_df,on='move_date',how='left').fillna(0)
    
    return agg_df

def load_intention_destination_df(raion_to_dest_df,all_raions,dest='PL'):
    good_sims = [5,6,7,9,11,12,14,19,21,23,26,29,39,40,43,45,46,48,49,50,51,53,54,63,67,78,86,87,98]
    #good_sims = [5]
    intention_sim = []

    dest_int = dest

    for sim in good_sims:
        #print(sim)

        cache_data = f'{CACHE_DIR}agg_intention_from_UKR_to_{dest_int}_sim_{str(sim).zfill(9)}.pq'

        if os.path.isfile(cache_data):
            intention_sim.append(pd.read_parquet(cache_data))
            continue

        all_dfs = []

        for raion in all_raions:
            #if raion!='Kyiv':
            #    continue
            #if raion=='Kyiv':
                #print(len(all_dfs))
            #print(raion)
            try:
                df_refugee = get_refugee_file(raion,str(sim).zfill(9))
            except:
                #print(raion,'intention was not simulated')
                continue
            df_refugee_with_dest = assign_destination(df_refugee,raion_to_dest_df,raion)
            agg_df_refugee = get_refugee_by_dest_agg(df_refugee_with_dest,raion)
            all_dfs.append(agg_df_refugee)
        #return all_dfs
        int_df = (pd.concat(all_dfs)).groupby('move_date')['h_size'].sum().reset_index()
        intention_sim.append(int_df)
        int_df.to_parquet(cache_data,index=False)
    return intention_sim


def load_intention_destination_df_by_demo_category(raion_to_dest_df,all_raions,dest='PL'):
    good_sims = [5,6,7,9,11,12,14,19,21,23,26,29,39,40,43,45,46,48,49,50,51,53,54,63,67,78,86,87,98]
    #good_sims = [5]
    intention_sim = []

    dest_int = dest

    for sim in good_sims:
        print(sim)

        cache_data = f'{CACHE_DIR}demo_aggv4_intention_from_UKR_to_{dest_int}_sim_{str(sim).zfill(9)}.pq'

        if os.path.isfile(cache_data):
            intention_sim.append(pd.read_parquet(cache_data))
            continue

        groups = ['single_male','single_female','male_only','female_only',
              'elderly_no_child','child_no_elderly','child_and_elderly','just_couples']
        all_dfs = []
        for raion in all_raions:
            #print(raion)
            #if raion!='Kyiv':
            #    continue
                #print(len(all_dfs))
            #print(raion)
            try:
                df_refugee = get_refugee_file(raion,str(sim).zfill(9))
            except:
                #print(raion,'intention was not simulated')
                continue
            df_refugee_with_dest = assign_destination(df_refugee,raion_to_dest_df,raion)
            agg_df_refugee = get_refugee_by_household_category_agg(df_refugee_with_dest,raion)
            #agg_df_refugee['matched'] =  agg_df_refugee[groups].sum(axis=1)
            all_dfs.append(agg_df_refugee)
            #break
            #if agg_df_refugee[agg_df_refugee.matched>1].shape[0]>0:
            #    break
        #return all_dfs
        int_df = (pd.concat(all_dfs)).groupby('move_date')[['h_size']+groups].sum().reset_index()
        intention_sim.append(int_df)
        int_df.to_parquet(cache_data,index=False)
    return intention_sim

def load_intention_destination_df_by_demo_category6(raion_to_dest_df,all_raions,dest='PL'):
    good_sims = [5,6,7,9,11,12,14,19,21,23,26,29,39,40,43,45,46,48,49,50,51,53,54,63,67,78,86,87,98]
    #good_sims = [5]
    intention_sim = []

    dest_int = dest

    for sim in good_sims:
        print(sim)

        cache_data = f'{CACHE_DIR}demo_cat6_aggv1_intention_from_UKR_to_{dest_int}_sim_{str(sim).zfill(9)}.pq'

        if os.path.isfile(cache_data):
            intention_sim.append(pd.read_parquet(cache_data))
            continue

        groups = ['male_only','female_only',
              'elderly_no_child','child_no_elderly',
              'child_and_elderly','just_couples']
        all_dfs = []
        for raion in all_raions:
            #print(raion)
            #if raion!='Kyiv':
            #    continue
                #print(len(all_dfs))
            #print(raion)
            try:
                df_refugee = get_refugee_file(raion,str(sim).zfill(9))
            except:
                #print(raion,'intention was not simulated')
                continue
            df_refugee_with_dest = assign_destination(df_refugee,raion_to_dest_df,raion)
            agg_df_refugee = get_refugee_by_household_category6_agg(df_refugee_with_dest,raion)
            #agg_df_refugee['matched'] =  agg_df_refugee[groups].sum(axis=1)
            all_dfs.append(agg_df_refugee)
            #break
            #if agg_df_refugee[agg_df_refugee.matched>1].shape[0]>0:
            #    break
        #return all_dfs
        int_df = (pd.concat(all_dfs)).groupby('move_date')[['h_size']+groups].sum().reset_index()
        intention_sim.append(int_df)
        int_df.to_parquet(cache_data,index=False)
    return intention_sim

def load_intention_destination_df_by_demo_category4(raion_to_dest_df,all_raions,dest='PL'):
    intention_sim_demo6 = load_intention_destination_df_by_demo_category6(raion_to_dest_df,all_raions)
    for i in range(0,len(intention_sim_demo6)):
        intention_sim_demo6[i]['elderly_and_or_child'] = intention_sim_demo6[i]['elderly_no_child']+intention_sim_demo6[i]['child_no_elderly']+intention_sim_demo6[i]['child_and_elderly']
        intention_sim_demo6[i] = intention_sim_demo6[i].drop(columns=['elderly_no_child','child_no_elderly','child_and_elderly'])
    
    return intention_sim_demo6

def get_conflict_data(ROLL,LAG):
    USE_NEIGHBOR = 5
    CONFLICT_DATA_PREFIX = 'ukraine_conflict_data_ADM2_HDX_buffer_'
    NEIGHBOR_DATA_PREFIX = 'ukraine_neighbor_'
    NETWORK_TYPE = '_R_0.01_P_0.04_Q_8_al_2.3.csv'
    NETWORK_TYPE_fast = '_R_0.01_P_0.04_Q_8_al_2.3.pq'
    
    total_impact_data = pd.read_csv(IMPACT_DIR+CONFLICT_DATA_PREFIX+str(USE_NEIGHBOR)+'_km.csv')
    total_impact_data['time'] = pd.to_datetime(total_impact_data['time'])

    conflict_context = total_impact_data.groupby('time')['event_intensity'].sum().reset_index()
    conflict_context = conflict_context.rename(columns={'event_intensity':'conflict'})

    conflict_context['conflict'] = conflict_context['conflict'].rolling(ROLL).mean()
    conflict_context = conflict_context.dropna(subset='conflict')
    NORM_CONSTANT = conflict_context['conflict'].max()
    conflict_context['conflict'] = conflict_context['conflict']/NORM_CONSTANT

    conflict_context['conflict'] = conflict_context['conflict'].shift(LAG)
    conflict_context = conflict_context.dropna(subset='conflict')

    conflict_context = conflict_context.rename(columns={'time':'move_date'})
    return conflict_context

def surrogate_conflict_haz(intention_sim_df,conflict_context,h,group_col='h_size',not_print_flag=0):
    if not not_print_flag:
        print('running surrogate conflict model with hazard',h)
    #conflict_context = get_conflict_data(roll,lag)
    intention_conflict_sim_df = intention_sim_df.merge(conflict_context,on='move_date',how='left').fillna(0)
    M = intention_conflict_sim_df[group_col].tolist() ## M(t): remaining migrants at time t
    C = intention_conflict_sim_df['conflict'].tolist() ## C(t): conflict context observed at time t
    R = []
    for sim in range(0,len(M)):
        R.append(0)
    for sim in range(0,len(M)):
        S = 1.0
        past_sim = sim
        while(past_sim>=0): ## people who migrated prior to this can return at current time
            #print('past sim:',past_sim)
            Returnee = M[past_sim]-M[past_sim]*S
            R[sim] = R[sim] + Returnee
            M[past_sim] = M[past_sim] - Returnee
            S = S*(1.0-h*(1-C[past_sim]))
            past_sim = past_sim - 1
    
    return_df = intention_sim_df.copy(deep=True)
    return_df['returnee'] = pd.Series(R)
    return return_df

def surrogate_conflict_haz_with_peer(intention_sim_df,conflict_context,h_up,h_down,thresh,not_print_flag=0):
    if not not_print_flag:
        print('running surrogate peer conflict model with hazard',h_up,'above',thresh,'peer and',h_down)
    #conflict_context = get_conflict_data(roll,lag)
    intention_conflict_sim_df = intention_sim_df.merge(conflict_context,on='move_date',how='left').fillna(0)
    M = intention_conflict_sim_df['h_size'].tolist() ## M(t): remaining migrants at time t
    C = intention_conflict_sim_df['conflict'].tolist() ## C(t): conflict context observed at time t
    R = []
    for sim in range(0,len(M)):
        R.append(0)
    tot_migrant = sum(M)
    cur_ret = 0.0
    for sim in range(0,len(M)):
        S = 1.0
        past_sim = sim
        peer_influence = cur_ret/tot_migrant
        if not not_print_flag:
            print('current peer pressure at sim time',sim,'is',peer_influence)
        h = h_up if (peer_influence)>=thresh else h_down
        while(past_sim>=0): ## people who migrated prior to this can return at current time
            #print('past sim:',past_sim)
            Returnee = M[past_sim]-M[past_sim]*S
            R[sim] = R[sim] + Returnee
            M[past_sim] = M[past_sim] - Returnee
            S = S*(1.0-h*(1-C[past_sim]))
            past_sim = past_sim - 1
        cur_ret = cur_ret + R[sim]
    return_df = intention_sim_df.copy(deep=True)
    return_df['returnee'] = pd.Series(R)
    return return_df

def surrogate_constant_haz(intention_sim_df,conflict_context,h,group_col='h_size',not_print_flag=0):
    if not not_print_flag:
        print('running surrogate model with hazard',h)
    M = intention_sim_df[group_col].tolist()
    R = []
    for sim in range(0,len(M)):
        R.append(0)
    for sim in range(0,len(M)):
        #print('sim:',sim)
        S = 1.0
        #R.append(0)
        past_sim = sim
        while(past_sim>=0):
            #print('past sim:',past_sim)
            Returnee = M[past_sim]-M[past_sim]*S
            R[sim] = R[sim] + Returnee
            M[past_sim] = M[past_sim] - Returnee
            S = S*(1.0-h)
            past_sim = past_sim - 1
            
    return_df = intention_sim_df.copy(deep=True)
    return_df['returnee'] = pd.Series(R)
    return return_df

def get_estimation_with_uncertainty(all_dfs,Q1=0.2,Q3=0.8,cache_clean_mode=0):
    return_ensemble_df = pd.concat(all_dfs)
    median_df = return_ensemble_df.groupby('move_date')['returnee'].quantile(q=0.5).reset_index()
    q1_df = (return_ensemble_df.groupby('move_date')['returnee'].quantile(q=Q1).reset_index()).rename(columns={'returnee':'q1'})
    q3_df = (return_ensemble_df.groupby('move_date')['returnee'].quantile(q=Q3).reset_index()).rename(columns={'returnee':'q3'})
    return_uncertain_df = (median_df.merge(q1_df,on='move_date',how='inner')).merge(q3_df,on='move_date',how='inner')
    return_uncertain_df = return_uncertain_df.rename(columns={'move_date':'return_date'})
    return return_uncertain_df

def get_estimation_with_uncertainty2(all_dfs,std_dev=2):
    return_ensemble_df = pd.concat(all_dfs)
    mean_df = return_ensemble_df.groupby('move_date')['returnee'].mean().reset_index()
    std_df = (return_ensemble_df.groupby('move_date')['returnee'].std().reset_index()).rename(columns={'returnee':'std'})
    return_uncertain_df = (mean_df.merge(std_df,on='move_date',how='inner'))
    return_uncertain_df['q1'] = return_uncertain_df['returnee']-return_uncertain_df['std']*std_dev
    return_uncertain_df['q3'] = return_uncertain_df['returnee']+return_uncertain_df['std']*std_dev
    return_uncertain_df = return_uncertain_df.rename(columns={'move_date':'return_date'})
    return return_uncertain_df

def compute_loss(sim_df, observed_df, observe_col, date_start, date_end,lerror=0.8, lcorr=0.2, date_col='return_date',gt_scale=1.0,print_flag=1):
    merged_df = sim_df.merge(observed_df,on=date_col,how='inner')
    merged_df = merged_df[(merged_df[date_col]>=pd.to_datetime(date_start)) & (merged_df[date_col]<=pd.to_datetime(date_end))]
    merged_df[observe_col] = merged_df[observe_col]/gt_scale
    mse = (((merged_df['returnee']-merged_df[observe_col])**2).sum())/merged_df.shape[0]
    corr = merged_df['returnee'].corr(merged_df[observe_col], method='pearson')
    rmse = mse**0.5
    nrmse = rmse/(max(merged_df[observe_col])-min(merged_df[observe_col]))
    if print_flag==1:
        print(nrmse,corr)    
    #if corr<0:
    #    nrmse = 2
    return lcorr*(1-corr)+lerror*nrmse,nrmse,corr

def compute_loss2(sim_df, observed_df, observe_col, date_start, date_end, date_col='return_date',gt_scale=1.0,print_flag=1):
    merged_df = sim_df.merge(observed_df,on=date_col,how='inner')
    merged_df = merged_df[(merged_df[date_col]>=pd.to_datetime(date_start)) & (merged_df[date_col]<=pd.to_datetime(date_end))]
    merged_df[observe_col] = merged_df[observe_col]/gt_scale
    mse = (((merged_df['returnee']-merged_df[observe_col])**2).sum())/merged_df.shape[0]
    corr = merged_df['returnee'].corr(merged_df[observe_col], method='pearson')
    rmse = mse**0.5
    nrmse = rmse/(max(merged_df[observe_col])-min(merged_df[observe_col]))
    return nrmse

def get_crps(all_sim_df,observed_df, observe_col, date_start, date_end, date_col='return_date',gt_scale=1.0):
    merged_df = all_sim_df.merge(observed_df,on=date_col,how='inner')
    merged_df = merged_df[(merged_df[date_col]>=pd.to_datetime(date_start)) & (merged_df[date_col]<=pd.to_datetime(date_end))]
    merged_df[observe_col] = merged_df[observe_col]/gt_scale
    y = merged_df.iloc[:,-1].values
    crps_en = ps.crps_ensemble(merged_df.iloc[:,-1].values,merged_df.iloc[:,1:-1].values)
    crps = crps_en.sum()/np.sum(np.absolute(y))
    return crps

def estimate_blackbox_function_loss(intention_sim,ukr_people_depart_poland_by_date,func,other_args,lcorr=0.2,lerr=0.8,
                                    date_start='2022-04-01',date_end='2022-08-01',no_print=0,q1=0.2,q3=0.8,conflict_context=None):    
    all_ret_df = []

    for idx,int_df in enumerate(intention_sim):
        ret_df = func(int_df,conflict_context,*other_args,not_print_flag=max(idx,no_print))
        ret_df['sim'] = str(idx)
        all_ret_df.append(ret_df)
    #print(all_ret_df[0].columns.tolist())
    crps = 0
    pivoted_df = (pd.concat(all_ret_df)).pivot(index='move_date',columns='sim',values='returnee').reset_index().rename(columns={'move_date':'return_date'})
    if not no_print:
        crps = get_crps(pivoted_df,ukr_people_depart_poland_by_date,'departure',date_start,date_end)
    final_ret_df = get_estimation_with_uncertainty(all_ret_df,Q1=q1,Q3=q3)
    ll,nrmse,corr = compute_loss(final_ret_df, ukr_people_depart_poland_by_date, 'departure', date_start, date_end,
                                 lerror=lerr, lcorr=lcorr,print_flag=1-no_print)
    #print('nrmse',nrmse,'corr',corr,'loss',ll)
    #final_ret_df = get_estimation_with_uncertainty2(all_ret_df,std_dev=1)
    #ll = compute_loss2(final_ret_df, ukr_people_depart_poland_by_date, 'departure', '2022-04-01', '2022-08-01',print_flag=0)
    #crps = 0
    #print('loss again',ll)
    return ll,nrmse,corr,crps,final_ret_df

def estimate_blackbox_function_loss_per_demo(intention_sim,ukr_people_depart_poland_by_date,func,group_args,
                                             date_start='2022-04-01',date_end='2022-08-01',no_print=0,q1=0.2,q3=0.8):
    groups = ['single_male','single_female','male_only','female_only',
              'elderly_no_child','child_no_elderly','child_and_elderly','just_couples']
    all_ret_df = []

    for idx,int_df in enumerate(intention_sim):
        group_ret_dfs = []
        for gidx,g in enumerate(groups):
            cur_ret_df = func(int_df,group_args[gidx],group_col=g,not_print_flag=max(idx,no_print))
            group_ret_dfs.append(cur_ret_df[['move_date','returnee']])
        ret_df = (pd.concat(group_ret_dfs)).groupby('move_date')['returnee'].sum().reset_index()
        ret_df['sim'] = str(idx)
        all_ret_df.append(ret_df)
    crps = 0
    pivoted_df = (pd.concat(all_ret_df)).pivot(index='move_date',columns='sim',values='returnee').reset_index().rename(columns={'move_date':'return_date'})
    if not no_print:
        crps = get_crps(pivoted_df,ukr_people_depart_poland_by_date,'departure',date_start,date_end)
    final_ret_df = get_estimation_with_uncertainty(all_ret_df,Q1=q1,Q3=q3)
    ll,nrmse,corr = compute_loss(final_ret_df, ukr_people_depart_poland_by_date, 'departure', date_start, date_end,print_flag=1-no_print)
    #print('nrmse',nrmse,'corr',corr,'loss',ll)
    #final_ret_df = get_estimation_with_uncertainty2(all_ret_df,std_dev=1)
    #ll = compute_loss2(final_ret_df, ukr_people_depart_poland_by_date, 'departure', '2022-04-01', '2022-08-01',print_flag=0)
    #crps = 0
    return ll,nrmse,corr,crps,final_ret_df

def estimate_blackbox_function_loss_per_demo6(intention_sim,ukr_people_depart_poland_by_date,func,group_args,lcorr=0.2,lerr=0.8,
                                              date_start='2022-04-01',date_end='2022-08-01',no_print=0,q1=0.2,q3=0.8):
    groups = ['male_only','female_only',
              'elderly_no_child','child_no_elderly','child_and_elderly','just_couples']
    all_ret_df = []

    for idx,int_df in enumerate(intention_sim):
        group_ret_dfs = []
        for gidx,g in enumerate(groups):
            cur_ret_df = func(int_df,group_args[gidx],group_col=g,not_print_flag=max(idx,no_print))
            group_ret_dfs.append(cur_ret_df[['move_date','returnee']])
        ret_df = (pd.concat(group_ret_dfs)).groupby('move_date')['returnee'].sum().reset_index()
        ret_df['sim'] = str(idx)
        all_ret_df.append(ret_df)
    crps = 0
    pivoted_df = (pd.concat(all_ret_df)).pivot(index='move_date',columns='sim',values='returnee').reset_index().rename(columns={'move_date':'return_date'})
    if not no_print:
        crps = get_crps(pivoted_df,ukr_people_depart_poland_by_date,'departure',date_start,date_end)
    final_ret_df = get_estimation_with_uncertainty(all_ret_df,Q1=q1,Q3=q3)
    ll,nrmse,corr = compute_loss(final_ret_df, ukr_people_depart_poland_by_date, 'departure', date_start, date_end,
                                 lerror=lerr, lcorr=lcorr,print_flag=1-no_print)
    #print('nrmse',nrmse,'corr',corr,'loss',ll)
    #final_ret_df = get_estimation_with_uncertainty2(all_ret_df,std_dev=1)
    #ll = compute_loss2(final_ret_df, ukr_people_depart_poland_by_date, 'departure', '2022-04-01', '2022-08-01',print_flag=0)
    #crps = 0
    return ll,nrmse,corr,crps,final_ret_df

def estimate_blackbox_function_loss_per_demo4(intention_sim,ukr_people_depart_poland_by_date,func,group_args,lcorr=0.2,lerr=0.8,
                                              date_start='2022-04-01',date_end='2022-08-01',no_print=0,q1=0.2,q3=0.8,conflict_context=None):
    groups = ['male_only','female_only', 'elderly_and_or_child','just_couples']
    all_ret_df = []

    for idx,int_df in enumerate(intention_sim):
        group_ret_dfs = []
        for gidx,g in enumerate(groups):
            cur_ret_df = func(int_df,conflict_context,group_args[gidx],group_col=g,not_print_flag=max(idx,no_print))
            group_ret_dfs.append(cur_ret_df[['move_date','returnee']])
        ret_df = (pd.concat(group_ret_dfs)).groupby('move_date')['returnee'].sum().reset_index()
        ret_df['sim'] = str(idx)
        all_ret_df.append(ret_df)
    crps = 0
    pivoted_df = (pd.concat(all_ret_df)).pivot(index='move_date',columns='sim',values='returnee').reset_index().rename(columns={'move_date':'return_date'})
    if not no_print:
        crps = get_crps(pivoted_df,ukr_people_depart_poland_by_date,'departure',date_start,date_end)
    final_ret_df = get_estimation_with_uncertainty(all_ret_df,Q1=q1,Q3=q3)
    ll,nrmse,corr = compute_loss(final_ret_df, ukr_people_depart_poland_by_date, 'departure', date_start, date_end,
                                 lerror=lerr, lcorr=lcorr,print_flag=1-no_print)
    #print('nrmse',nrmse,'corr',corr,'loss',ll)
    #final_ret_df = get_estimation_with_uncertainty2(all_ret_df,std_dev=1)
    #ll = compute_loss2(final_ret_df, ukr_people_depart_poland_by_date, 'departure', '2022-04-01', '2022-08-01',print_flag=0)
    #crps = 0
    return ll,nrmse,corr,crps,final_ret_df

def plot_sim_and_gt(ax,sim_df,model_name,gt_df,lgd_params,conflict_data=None,gt_scale=1,
                    country_name='PL',date_start='2022-04-01',date_end='2022-08-01',top_k=1,not_plot_gt=0,coverage=0.5):
    #print('visualization for',model_name)
    base_color = '#d95f0e'
    cover_alpha_dict = {0.5:0.4,0.9:0.3,0.95:0.2}
    alpha = cover_alpha_dict[coverage] if coverage in cover_alpha_dict else 0.3
    ax.plot(sim_df['return_date'],sim_df['returnee']/1000,alpha=1.0/top_k,color=base_color,linewidth=3)
    ax.fill_between(sim_df['return_date'],sim_df['q1']/1000,sim_df['q3']/1000,alpha=alpha/top_k,color=base_color,label=str(int(coverage*100))+'% CI')
    if not not_plot_gt:
        ax.scatter(gt_df['return_date'][::7],gt_df['departure'][::7]/(1000*gt_scale),facecolor='none',edgecolor='black',s=200)
    
    if conflict_data is not None:
        print('plotting conflict data')
        ax2 = ax.twinx()
        ax2.plot(conflict_context['time'],conflict_context['conflict'],label='C(t)',color='red')
            
    #ll,err,corr = compute_loss(sim_df, gt_df, 'departure',date_start,date_end,gt_scale=gt_scale)
    
    myFmt = mdates.DateFormatter('%m-%d')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0,interval=4))
    ax.xaxis.set_major_formatter(myFmt)
    ax.set_xlim([pd.to_datetime(date_start),pd.to_datetime(date_end)])
    ax.legend(loc="lower left", fancybox=True, handlelength=lgd_params['hlen'], borderpad=lgd_params['bpad'], labelspacing=lgd_params['lspace'],
                 handletextpad = lgd_params['htxtpad'], borderaxespad = lgd_params['baxpad'], columnspacing = lgd_params['cspace'],
                 ncol=lgd_params['ncol'], edgecolor=lgd_params['ecolor'], frameon=True, framealpha=lgd_params['alpha'], shadow=True, 
                 prop={'size': lgd_params['size']})
    ax.set_ylabel('Returnee (K)')
    #ax.set_title(f'NRMSE: {round(err,4)}\nPCC: {round(corr,4)}')
    ax.set_xlabel('Date')
    
    props = {"rotation" : 0}
    plt.setp(ax.get_xticklabels(), **props)
    
def plot_error(ax,sim_df,model_name,gt_df,lgd_params,gt_scale=1,add_error=0,
                    country_name='PL',date_start='2022-04-01',date_end='2022-08-01'):
        
    ll,err,corr = compute_loss(sim_df, gt_df, 'departure',date_start,date_end,gt_scale=gt_scale)
    print('corr:',corr)
    print('nrmse:',err)

    handles, labels = ax.get_legend_handles_labels()
    
    err_patch = mlines.Line2D([],[],color='#8c510a', label='NRMSE: '+str(round(err,4)),linestyle='None',markersize=20,marker='v') 
    #mpatches.Patch(color='#ec7014', label='NRMSE: '+str(round(err,4)))
    corr_patch = mlines.Line2D([],[],color='#01665e', label='PCC: '+str(round(corr,4)),linestyle='None',marker='^',markersize=20)
    #mpatches.Patch(color='#41ab5d', label='PCC: '+str(round(corr,4)),marker='^')
    patchlist = handles+[err_patch,corr_patch]
    
    ax.legend(handles=patchlist,loc="best", fancybox=True, handlelength=lgd_params['hlen'], borderpad=lgd_params['bpad'], labelspacing=lgd_params['lspace'],
                 handletextpad = lgd_params['htxtpad'], borderaxespad = lgd_params['baxpad'], columnspacing = lgd_params['cspace'],
                 ncol=lgd_params['ncol'], edgecolor=lgd_params['ecolor'], frameon=True, framealpha=lgd_params['alpha'], shadow=True, 
                 prop={'size': lgd_params['size']})
    return corr,err

def get_CI_coverage(sim_df, observed_df, observe_col, date_start, date_end, date_col='return_date',gt_scale=1.0):
    merged_df = sim_df.merge(observed_df,on=date_col,how='inner')
    merged_df = merged_df[(merged_df[date_col]>=pd.to_datetime(date_start)) & (merged_df[date_col]<=pd.to_datetime(date_end))]
    merged_df[observe_col] = merged_df[observe_col]/gt_scale
    merged_df['covered'] = merged_df.apply(lambda x: 1 if (x[observe_col]>=x['q1']) and (x[observe_col]<=x['q3']) else 0, axis=1)
    return merged_df['covered'].sum()/merged_df.shape[0]
    #return merg
