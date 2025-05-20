START_DATE = '2022-02-24'
END_DATE = '2022-09-01'
T_CURRENT = pd.to_datetime(START_DATE)
T_FINAL = pd.to_datetime(END_DATE)
NO_RETURN_DATE = pd.to_datetime('2025-01-01')

if 'global_conflict' not in function_type:
    impact_data = total_impact_data[total_impact_data.matching_place_id==raion_name]
else:
    print('looking at whole UKR')
    impact_data = total_impact_data

impact_data['time'] = pd.to_datetime(impact_data['time'])
if 'intensity' not in function_type:
    print('working based on count')
    conflict_context = impact_data.groupby('time')['event_id'].count().reset_index()
    conflict_context = conflict_context.rename(columns={'event_id':'conflict'})
else:
    print('working based on intensity')
    conflict_context = impact_data.groupby('time')['event_intensity'].sum().reset_index()
    conflict_context = conflict_context.rename(columns={'event_intensity':'conflict'})

NORM_CONSTANT = conflict_context['conflict'].max()
conflict_context['conflict'] = conflict_context['conflict'].rolling(14).mean()
conflict_context = conflict_context.dropna(subset='conflict')
conflict_context['conflict'] = conflict_context['conflict']/NORM_CONSTANT

df_refugee['move_date'] = pd.to_datetime(df_refugee['move_date'])
df_refugee['return_date'] = pd.to_datetime('2025-01-01')
df_refugee['demo_move_prob'] = df_refugee.apply(lambda x: get_return_prob_demographic(x[DEMO_TYPES].tolist(),FAMILY_PROB,MOVE_PROB),axis=1)
df_refugee['survival_prob'] = 1.0

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