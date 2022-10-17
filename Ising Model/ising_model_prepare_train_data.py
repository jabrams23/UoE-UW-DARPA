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





def main(params):
    
    order_param = params['order_param']
    mask_type = params['mask_type'] 
    smoothing = params['smoothing']
    target_duration = params['target_duration']
    
    if smoothing == 'gaussian':
        # smooth_params = [[24,0],[48,0]]
        # smooth_params = [[48,0]]
        smooth_params = [[96,0]]
    else:
        smooth_params = [[0,0]]
        
    # data_type = 'raw'
    data_type = 'EWS'
    
    
    include_reverse_time = 0
    
    if mask_type is None:
        out_dir = os.path.join('Ising_Output','var_'+order_param)
    else:
        out_dir = os.path.join('Ising_Output','var_'+order_param+'_'+mask_type)
    
    
    
    for smooth_param_ind,smooth_param in enumerate(smooth_params):
        
        
        subdirs = ['train','test','validate']
        train_classes = [0,1,2]
        class_dfs = []
        for sj,subdir in enumerate(subdirs):
            if smoothing == None:
                data_dir = os.path.join(out_dir,subdir,'Processed',data_type)
            elif smoothing == 'gaussian':
                data_dir = os.path.join(out_dir,subdir,'Processed','Gaussian_{}_{}'.format(smooth_param[0],smooth_param[1]),data_type)
                
            if not os.path.exists(data_dir):
                print('Skipping smoothing parameter ' + str(smooth_param) + ': Processed data files not present')
                continue
            file_list = glob.glob(os.path.join(data_dir,'*.pkl'))
            # file_list = glob.glob(os.path.join(data_dir,'*.npz'))
            
            # train_classes = np.random.choice([0,1,2],size=len(file_list),p=[0.8,0.1,0.1]) # train, test, validate
            
            df_cols = list(pd.read_pickle(file_list[0]).columns)
            df_cols.append('train_class')
            
            this_class_df = pd.DataFrame(columns=df_cols)
            
            for fj, fname in enumerate(file_list):
                this_df = pd.read_pickle(fname)
                this_df['train_class'] = train_classes[sj]
                this_class_df = pd.concat([this_class_df,this_df],ignore_index=True,sort=False)
            class_dfs.append(this_class_df)
        
        data_df = pd.concat(class_dfs)
        
        all_s = [xj for xj in np.array(data_df['x'])]
        all_null = np.array([xj for xj in data_df['null']])
        all_train_classes = np.array([xj for xj in data_df['train_class']])
        all_time_dir = np.array([xj for xj in data_df['time_dir']])
        all_t_roll_window = np.array([xj for xj in data_df['t_roll_window']])
        try:
            all_Tbounds = np.array([xj for xj in data_df['Tbounds']])
        except KeyError:
            print('Tbounds missing')
            all_Tbounds = np.nan*all_null
        # s_list = []
        # null_list = []
        # time_dir_list = []
        # t_roll_window_list = []
        # train_classes_list = []
        
        # train_classes = np.random.choice([0,1,2],size=len(file_list),p=[0.8,0.1,0.1]) # train, test, validate
        
        # for jf, fname in enumerate(file_list):
        #     with np.load(fname, allow_pickle=True) as np_load:
        #         s = np_load['s']
        #         null = np_load['null']
        #         time_dir = np_load['time_dir']
        #         t_roll_window = np_load['t_roll_window']
        #     s_list.append(s)
        #     null_list.append(null)
        #     time_dir_list.append(time_dir)
        #     t_roll_window_list.append(t_roll_window)
        #     train_classes_list.append(np.array([train_classes[jf] for _ in null]))
        
        # all_s = np.concatenate(s_list,axis=0)
        # all_null = np.concatenate(null_list,axis=0)
        # all_time_dir = np.concatenate(time_dir_list,axis=0)
        # all_t_roll_window = np.concatenate(t_roll_window_list,axis=0)
        # all_train_classes = np.concatenate(train_classes_list,axis=0)
        
        if data_type == 'EWS':
            all_s = np.array(all_s)
            nan_mask = ~np.any(np.isnan(all_s),axis=(1,2)) # omit runs with NaN values
            print('Omitting ' + str(np.count_nonzero(nan_mask==0)) + ' runs which contain NaN values')
            all_s = all_s[nan_mask,:,:]
            all_null = all_null[nan_mask]
            all_train_classes = all_train_classes[nan_mask]
            all_time_dir = all_time_dir[nan_mask]
            all_t_roll_window = all_t_roll_window[nan_mask]
            all_Tbounds = all_Tbounds[nan_mask]
            
            
        elif data_type == 'raw':
            all_s_pad = np.zeros((len(all_s),target_duration,all_s[0].shape[1],all_s[0].shape[2]))
            for j, sj in enumerate(all_s):
                all_s_pad[j,-sj.shape[0]:,:,:] = sj
            all_s = all_s_pad
        
        if include_reverse_time == 0:
            if data_type == 'EWS':
                all_s = all_s[all_time_dir == 1,:,:]
            elif data_type == 'raw':
                all_s = all_s[all_time_dir == 1,:,:,:]
            all_null = all_null[all_time_dir == 1]
            all_train_classes = all_train_classes[all_time_dir == 1]
            all_t_roll_window = all_t_roll_window[all_time_dir == 1]
            all_time_dir = all_time_dir[all_time_dir == 1]
            all_Tbounds = all_Tbounds[all_time_dir == 1]
        
        print('Balancing {} Null runs and {} Trans. runs'.format(np.count_nonzero(all_null==1),np.count_nonzero(all_null==0)))
        
        if np.count_nonzero(all_null==1) > np.count_nonzero(all_null==0):
            excess_inds = all_null==1
        else:
            excess_inds = all_null==0
        
        target_class_size = np.min([np.count_nonzero(all_null==1),np.count_nonzero(all_null==0)])
        n_delete = np.count_nonzero(excess_inds) - target_class_size
        delete_inds = np.where(excess_inds)[0]
        delete_inds = np.random.choice(delete_inds,size=n_delete,replace=False)
        
        all_s = np.delete(all_s,delete_inds,axis=0)
        all_null = np.delete(all_null,delete_inds)
        all_train_classes = np.delete(all_train_classes,delete_inds)
        all_time_dir = np.delete(all_time_dir,delete_inds)
        all_t_roll_window = np.delete(all_t_roll_window,delete_inds)
        all_Tbounds = np.delete(all_Tbounds,delete_inds)
        
        print('Outputting {} Null runs and {} Trans. runs'.format(np.count_nonzero(all_null==1),np.count_nonzero(all_null==0)))
    
        
        
        out_dict = {'null':all_null,
                    's':all_s,
                    'train_classes':all_train_classes,
                    'time_dir':all_time_dir,
                    't_roll_window':all_t_roll_window,
                    'Tbounds':all_Tbounds
                    }
        
        
        outfile_name = 'train_data_gaussian_{}_{}.npz'.format(smooth_param[0],smooth_param[1])
        
        
        print('Saving to disk: ' + os.path.join(out_dir,outfile_name))
        
        np.savez_compressed(os.path.join(out_dir,outfile_name),**out_dict,
                            allow_pickle=True, fix_imports=True)
        
if __name__ == "__main__":
    params = {}
    # params['order_param'] = 'h'
    # params['order_param'] = 'h_lin'
    # params['order_param'] = 'temp'
    params['order_param'] = 'temp_lin'
    # params['order_param'] = 'temp_local'

    params['mask_type'] = None
    # params['mask_type'] = 'ellipse'
    
    # params['smoothing'] = None
    params['smoothing'] = 'gaussian'

    params['target_duration'] = 600
    
    main(params)