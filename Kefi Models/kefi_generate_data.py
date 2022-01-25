# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from kefi_et_al_models import *
import os
import glob


n_runs = 100
which_model = 1 - 1 # -1 to offset 0 indexing
output_burn = True
plot_res = True

out_dir = os.path.join('Kefi_Models')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
m1_consts = {
    'D':0.05,
    'lamb':0.12,
    'rho':1,
    'B_c':1,
    'mu':2,
    'B_0':1,
    # 'sigma_w':0.1,
    # 'sigma_B':0.25
    'sigma_w':0.1*0.25,
    'sigma_B':0.25*0.25
    }

m2_consts = {
    'm':0.1,
    'f':0.9,
    'delta':0.1,
    'g':31,
    'c':0.3,
    'r':0.01,
    'd':0.2
    }

m3_consts = {
    'c':10,
    'g_max':0.05,
    'k_1':5,
    'd':0.25,
    'a':0.2,
    'k_2':5,
    'W_0':0.2,
    'r_w':0.2,
    'D_p':0.1,
    'D_w':0.1,
    'D_o':100,
    'sigma':0.001 #0.1*0.25 # the reference paper fails to provide a value for sigma
    }

all_consts = [m1_consts,m2_consts,m3_consts]
all_gen_fn = [generate_model1, generate_model2, generate_model3]
# all_alpha_crit = [1.936,0,0.875]
all_run_time = [1200,800,600]

m1_cols = ['w','B']
m2_cols = ['w']
m3_cols = ['O','W','P']
all_cols = [m1_cols,m2_cols,m3_cols]

consts = all_consts[which_model]
run_time = all_run_time[which_model]
gen_fn = all_gen_fn[which_model]
# alpha_crit = all_alpha_crit[which_model]


existing_files = glob.glob(os.path.join(out_dir,'Model_'+ str(which_model+1)+'_*'))
if len(existing_files) > 0:
    existing_file_inds = [int(os.path.split(fn)[-1][-7:-4]) for fn in existing_files]
    start_file_ind = int(np.max(existing_file_inds))+1
else:
    start_file_ind = 0
for rj in range(start_file_ind,start_file_ind+n_runs): 
    # this_null = np.random.choice([0,1])
    this_null = 0
    if this_null == 1:
        this_run_time = int(run_time/2)
    else:
        this_run_time = run_time
    
    # if this_null:
    #     this_offset = np.random.choice([-1,1])
    #     this_alpha_interval = alpha_crit + this_offset*alpha_crit*np.array([np.random.uniform(0.1,0.3),np.random.uniform(0.3,0.4)])
    # else:
    #     this_alpha_interval = np.random.permutation(alpha_crit*(1+np.array([-np.random.uniform(0.2,0.5),np.random.uniform(0.2,0.5)])))
        
    if which_model == 1: # model 2
        this_alpha_interval = [0.3,2]
    elif which_model == 2: # model 3
        this_alpha_interval = [2,0.05]
    # this_alpha_interval = [2,0.1]
    
    this_row = gen_fn(consts,this_alpha_interval,rj,run_time=this_run_time,plot_res=plot_res,return_burn=True)

    
    out_df = pd.DataFrame(this_row)
    out_df['null'] = [this_null]
    out_df.to_pickle(os.path.join(out_dir,'Model_'+ str(which_model+1)+'_Output_{:03d}.pkl'.format(rj)))
