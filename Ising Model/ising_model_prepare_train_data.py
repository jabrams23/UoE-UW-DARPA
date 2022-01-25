# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:29:59 2021

@author: Daniel Dylewsky

Prepare processed Ising model data for use in training of neural network
Consolidate runs into Numpy format, separate into train/test/validate classes
"""

import numpy as np
import os
import glob
import pandas as pd


# order_param = 'h'
# order_param = 'h_lin'
# order_param = 'temp'
order_param = 'temp_lin'

# smoothing = None
smoothing = 'gaussian'

include_reverse_time = 0

if smoothing == 'gaussian':
    # smooth_params = [[24,0],[48,0],[96,0]]
    smooth_params = [[96,0]]
else:
    smooth_params = [[0,0]]

# mask_type = None
mask_type = 'ellipse'

if mask_type is None:
    out_dir = os.path.join('Ising_Output','var_'+order_param)
else:
    out_dir = os.path.join('Ising_Output','var_'+order_param+'_'+mask_type)



for smooth_param_ind,smooth_param in enumerate(smooth_params):
    
    if smoothing == None:
        data_dir = os.path.join(out_dir,'Processed')
    elif smoothing == 'gaussian':
        data_dir = os.path.join(out_dir,'Processed','Gaussian_{}_{}'.format(smooth_param[0],smooth_param[1]))
        
    if not os.path.exists(data_dir):
        print('Skipping smoothing parameter ' + str(smooth_param) + ': Processed data files not present')
        continue
    
    file_list = glob.glob(os.path.join(data_dir,'*.pkl'))
    train_classes = np.random.choice([0,1,2],size=len(file_list),p=[0.8,0.1,0.1]) # train, test, validate
    df_cols = list(pd.read_pickle(file_list[0]).columns)
    df_cols.append('train_class')
    
    data_df = pd.DataFrame(columns=df_cols)
    
    for fj, fname in enumerate(file_list):
        this_df = pd.read_pickle(fname)
        this_df['train_class'] = train_classes[fj]
        data_df = pd.concat([data_df,this_df],ignore_index=True)
        
        
    all_s = np.array([xj for xj in data_df['x']])
    all_null = np.array([xj for xj in data_df['null']])
    all_train_classes = np.array([xj for xj in data_df['train_class']])
    all_time_dir = np.array([xj for xj in data_df['time_dir']])
    all_t_roll_window = np.array([xj for xj in data_df['t_roll_window']])
    # data_df.to_pickle(os.path.join(data_dir,'train_data.pkl'))
    
    nan_mask = ~np.any(np.isnan(all_s),axis=(1,2)) # omit runs with NaN values
    print('Omitting ' + str(np.count_nonzero(nan_mask==0)) + ' runs which contain NaN values')
    all_s = all_s[nan_mask,:,:]
    all_null = all_null[nan_mask]
    all_train_classes = all_train_classes[nan_mask]
    all_time_dir = all_time_dir[nan_mask]
    all_t_roll_window = all_t_roll_window[nan_mask]
    
    if include_reverse_time == 0:
        all_s = all_s[all_time_dir == 1,:,:]
        all_null = all_null[all_time_dir == 1]
        all_train_classes = all_train_classes[all_time_dir == 1]
        all_t_roll_window = all_t_roll_window[all_time_dir == 1]
        all_time_dir = all_time_dir[all_time_dir == 1]
    
    out_dict = {'null':all_null,
                's':all_s,
                'train_classes':all_train_classes,
                'time_dir':all_time_dir,
                't_roll_window':all_t_roll_window
                }
    print('Saving to disk: ' + os.path.join(data_dir,'train_data.npz'))
    np.savez_compressed(os.path.join(data_dir,'train_data.npz'),**out_dict,
                        allow_pickle=True, fix_imports=True)