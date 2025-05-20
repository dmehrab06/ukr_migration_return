from file_paths_and_consts import *
import pandas as pd
import os
import matplotlib.pyplot as plt
import heapq
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import properscoring as ps
import numpy as np
import matplotlib.dates as mdates

def get_return_df(df):
    df['Total'] = df[DEMO_TYPES].sum(axis=1)
    df['return_date'] = pd.to_datetime(df['return_date'])
    return_df = df.groupby('return_date')['Total'].sum().reset_index()
    return_df['Total'] = return_df['Total'].rolling(7).mean()
    return_df = return_df.dropna(subset=['Total'])
    return return_df

def get_all_estimation(param,return_dir_prefix,all_raions,dest_name,cache_clean_mode=0):
    RETURN_DIR = f'{OUTPUT_DIR}{return_dir_prefix}_{param}/'
    cache_file_name = f'calib_return_all_est_from_{dest_name}_to_UKR_method_{return_dir_prefix}_from_{param}.pq'
    if os.path.isfile(CACHE_DIR+cache_file_name) and cache_clean_mode==0:
        #print('cache found for',RETURN_DIR,flush=True)
        df = pd.read_parquet(CACHE_DIR+cache_file_name)
        df['return_date'] = pd.to_datetime(df['return_date'])
        df = df.fillna(0)
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
        #print('try to find either',RETURN_DIR+base_file,'or',RETURN_DIR+base_file_fast)
        if os.path.isfile(RETURN_DIR+base_file) or os.path.isfile(RETURN_DIR+base_file_fast):
            #print('found either',RETURN_DIR+base_file,'or',RETURN_DIR+base_file_fast)
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
                all_return_df['sim'] = sim_idx
                single_sim_return_df.append(all_return_df)
    return_ensemble_df = (pd.concat(single_sim_return_df)).pivot(index='return_date',columns='sim',values='Total').reset_index()
    return_ensemble_df.to_parquet(CACHE_DIR+cache_file_name,index=False)
    return return_ensemble_df

def get_estimation(param,return_dir_prefix,all_raions,dest_name,Q1=0.2,Q3=0.8,cache_clean_mode=0):
    print(Q1,Q3)
    RETURN_DIR = f'{OUTPUT_DIR}{return_dir_prefix}_{param}/'
    cache_file_name = f'calib_return_est_from_{dest_name}_to_UKR_method_{return_dir_prefix}_from_{param}.pq'
    if os.path.isfile(CACHE_DIR+cache_file_name) and cache_clean_mode==0:
        #print('cache found for',RETURN_DIR,flush=True)
        df = pd.read_parquet(CACHE_DIR+cache_file_name)
        df['return_date'] = pd.to_datetime(df['return_date'])
        df = df.fillna(0)
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
        #print('try to find either',RETURN_DIR+base_file,'or',RETURN_DIR+base_file_fast)
        if os.path.isfile(RETURN_DIR+base_file) or os.path.isfile(RETURN_DIR+base_file_fast):
            #print('found either',RETURN_DIR+base_file,'or',RETURN_DIR+base_file_fast)
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
    return return_uncertain_df

def get_estimation_Q1_Q3(param,return_dir_prefix,all_raions,dest_name,alpha=0.6,cache_clean_mode=0):
    Q1 = 0.5-alpha/2.0
    Q3 = 0.5+alpha/2.0
    RETURN_DIR = f'{OUTPUT_DIR}{return_dir_prefix}_{param}/'
    cache_file_name = f'calib_return_est_from_{dest_name}_to_UKR_method_{return_dir_prefix}_from_{param}_quantile_{alpha}.pq'
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
        #print('try to find either',RETURN_DIR+base_file,'or',RETURN_DIR+base_file_fast)
        if os.path.isfile(RETURN_DIR+base_file) or os.path.isfile(RETURN_DIR+base_file_fast):
            #print('found either',RETURN_DIR+base_file,'or',RETURN_DIR+base_file_fast)
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
    return return_uncertain_df

def get_estimation_for_raion(param,return_dir_prefix,single_raion,dest_name,Q1=0.2,Q3=0.8):
    RETURN_DIR = f'{OUTPUT_DIR}{return_dir_prefix}_{param}/'
    #print('return dir is',RETURN_DIR,flush=True)
    single_sim_return_df = []
    for sim in [11]:
        sim_idx = str(sim).zfill(9)
        base_file = f'mim_hid_return_Kyiv_SIM_{sim_idx}_SEED_0.csv'
        base_file_fast = f'mim_hid_return_Kyiv_SIM_{sim_idx}_SEED_0.pq'
        #print('base_file_loc',RETURN_DIR+base_file,'khujtesi')
        if os.path.isfile(RETURN_DIR+base_file) or os.path.isfile(RETURN_DIR+base_file_fast):
            cur_settings_df = []
            for raion_name in [single_raion]:
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
    #return_uncertain_df.to_parquet(CACHE_DIR+cache_file_name,index=False)
    return return_uncertain_df

def compute_loss_ABM(sim_df, observed_df, observe_col, date_start, date_end, date_col='return_date',gt_scale=1.0):
    merged_df = sim_df.merge(observed_df,on=date_col,how='inner')
    merged_df = merged_df[(merged_df[date_col]>=pd.to_datetime(date_start)) & (merged_df[date_col]<=pd.to_datetime(date_end))]
    merged_df[observe_col] = merged_df[observe_col]/gt_scale
    mse = (((merged_df['Total']-merged_df[observe_col])**2).sum())/merged_df.shape[0]
    corr = merged_df['Total'].corr(merged_df[observe_col], method='pearson')
    rmse = mse**0.5
    nrmse = rmse/(max(merged_df[observe_col])-min(merged_df[observe_col]))
    #print(nrmse,corr)
    if corr<0:
        #return max(merged_df[observe_col])*10
        nrmse = 2
    return 0.2*(1-corr)+0.8*nrmse,corr,nrmse
    #return merged_df, corr,nrmse
    
def get_opt_scale(sim_df, observed_df, observe_col, date_start, date_end, date_col='return_date'):
    merged_df = sim_df.merge(observed_df,on=date_col,how='inner')
    merged_df = merged_df[(merged_df[date_col]>=pd.to_datetime(date_start)) & (merged_df[date_col]<=pd.to_datetime(date_end))]
    denom = sum(merged_df['Total']**2)
    nom = sum(merged_df['Total']*merged_df[observe_col])
    return nom/denom
    #return merged_df, corr,nrmse
    
def plot_sim_and_gt(ax,sim_df,model_name,gt_df,lgd_params,conflict_data=None,gt_scale=1,add_error=0,
                    country_name='PL',date_start='2022-04-01',date_end='2022-08-01',top_k=1,not_plot_gt=0,coverage=0.5):
    #print('visualization for',model_name)
    base_color = '#08519c'
    cover_alpha_dict = {0.5:0.4,0.9:0.3,0.95:0.2}
    alpha = cover_alpha_dict[coverage] if coverage in cover_alpha_dict else 0.3
    ax.plot(sim_df['return_date'],sim_df['Total']/1000,alpha=1.0/top_k,color=base_color,linewidth=3)
    ax.fill_between(sim_df['return_date'],sim_df['q1']/1000,sim_df['q3']/1000,alpha=alpha/top_k,color=base_color,label=str(int(coverage*100))+'% CI')
    if not not_plot_gt:
        ax.scatter(gt_df['return_date'][::7],gt_df['departure'][::7]/(1000*gt_scale),facecolor='none',edgecolor='black',s=200)
    
    if conflict_data is not None:
        print('plotting conflict data')
        ax2 = ax.twinx()
        ax2.plot(conflict_context['time'],conflict_context['conflict'],label='C(t)',color='red')
        
    ll,corr,err = compute_loss_ABM(sim_df, gt_df, 'departure',date_start,date_end,gt_scale=gt_scale)

    handles, labels = ax.get_legend_handles_labels()
    #print(labels)
    err_patch = mpatches.Patch(color='#ec7014', label='NRMSE: '+str(round(err,4)))
    corr_patch = mpatches.Patch(color='#41ab5d', label='PCC: '+str(round(corr,4)))
    
    if add_error:
        handles.extend([err_patch,corr_patch])
        #handles.append(corr_patch)
        handles, labels = ax.get_legend_handles_labels()
        #print('after error adding')
        #print(labels)
    
    myFmt = mdates.DateFormatter('%m/%d')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0,interval=4))
    ax.xaxis.set_major_formatter(myFmt)
    ax.set_xlim([pd.to_datetime(date_start),pd.to_datetime(date_end)])
    ax.legend(loc="best", fancybox=True, handlelength=lgd_params['hlen'], borderpad=lgd_params['bpad'], labelspacing=lgd_params['lspace'],
                 handletextpad = lgd_params['htxtpad'], borderaxespad = lgd_params['baxpad'], columnspacing = lgd_params['cspace'],
                 ncol=lgd_params['ncol'], edgecolor=lgd_params['ecolor'], frameon=True, framealpha=lgd_params['alpha'], shadow=True, 
                 prop={'size': lgd_params['size']})
    ax.set_ylabel('Returnee (K)')
    #ax.set_title(model_name)
    ax.set_xlabel('Date')
    
    props = {"rotation" : 0}
    plt.setp(ax.get_xticklabels(), **props)
    
def plot_error(ax,sim_df,model_name,gt_df,lgd_params,gt_scale=1,add_error=0,
                    country_name='PL',date_start='2022-04-01',date_end='2022-08-01'):
        
    ll,corr,err = compute_loss_ABM(sim_df, gt_df, 'departure',date_start,date_end,gt_scale=gt_scale)
    print('corr:',corr)
    print('nrmse:',err)

    handles, labels = ax.get_legend_handles_labels()
    
    err_patch = mlines.Line2D([],[],color='#ec7014', label='NRMSE: '+str(round(err,4)),linestyle='None',markersize=20,marker='v') 
    #mpatches.Patch(color='#ec7014', label='NRMSE: '+str(round(err,4)))
    corr_patch = mlines.Line2D([],[],color='#41ab5d', label='PCC: '+str(round(corr,4)),linestyle='None',marker='^',markersize=20)
    #mpatches.Patch(color='#41ab5d', label='PCC: '+str(round(corr,4)),marker='^')
    patchlist = handles+[err_patch,corr_patch]
    
    ax.legend(handles=patchlist,loc="best", fancybox=True, handlelength=lgd_params['hlen'], borderpad=lgd_params['bpad'], labelspacing=lgd_params['lspace'],
                 handletextpad = lgd_params['htxtpad'], borderaxespad = lgd_params['baxpad'], columnspacing = lgd_params['cspace'],
                 ncol=lgd_params['ncol'], edgecolor=lgd_params['ecolor'], frameon=True, framealpha=lgd_params['alpha'], shadow=True, 
                 prop={'size': lgd_params['size']})
    return corr,err
    
def findminobservation_idx(xs,ys,k=1):
    return [idx for idx, _ in heapq.nsmallest(k, enumerate(ys), key=lambda x: x[1])]

def get_CI_coverage(sim_df, observed_df, observe_col, date_start, date_end, date_col='return_date',gt_scale=1.0):
    merged_df = sim_df.merge(observed_df,on=date_col,how='inner')
    merged_df = merged_df[(merged_df[date_col]>=pd.to_datetime(date_start)) & (merged_df[date_col]<=pd.to_datetime(date_end))]
    merged_df[observe_col] = merged_df[observe_col]/gt_scale
    merged_df['covered'] = merged_df.apply(lambda x: 1 if (x[observe_col]>=x['q1']) and (x[observe_col]<=x['q3']) else 0, axis=1)
    return merged_df['covered'].sum()/merged_df.shape[0]
    #return merged_df
        
def get_crps(all_sim_df,observed_df, observe_col, date_start, date_end, date_col='return_date',gt_scale=1.0):
    merged_df = all_sim_df.merge(observed_df,on=date_col,how='inner')
    merged_df = merged_df[(merged_df[date_col]>=pd.to_datetime(date_start)) & (merged_df[date_col]<=pd.to_datetime(date_end))]
    merged_df[observe_col] = merged_df[observe_col]/gt_scale
    y = merged_df.iloc[:,-1].values
    crps_en = ps.crps_ensemble(merged_df.iloc[:,-1].values,merged_df.iloc[:,1:-1].values)
    crps = crps_en.sum()/np.sum(np.absolute(y))
    return crps