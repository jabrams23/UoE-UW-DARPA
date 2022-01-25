# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:29:59 2021

@author: Daniel Dylewsky

Animate simulation results from kefi_generate_data.py and output to .mp4 file
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
import pandas as pd

def animate(i):
    img.set_data(s[i,:,:])
    this_alpha = alpha_grid[i]
    ttl.set_text(run_id + '(null='+str(null)+')\nvar = '+var+', alpha =  {0:.2f}'.format(this_alpha))
    

which_model = 2


data_dir = 'Kefi_Models'



vid_dir = os.path.join(data_dir,'Video')
if not os.path.exists(vid_dir):
    os.makedirs(vid_dir)
    
var_names_dict = {1:['w','B'],
                  2:['w'],
              3:['O','W','P']}

var_names = var_names_dict[which_model]

file_list = np.array(glob.glob(os.path.join(data_dir,'Model_'+str(which_model)+'*.pkl')))

for fname in file_list:
    print('Generating animation for ' + fname)
    run_id = os.path.split(fname)[-1][:-4]
    this_df = pd.read_pickle(fname)
    alpha_interval = this_df['alpha_interval'].values[0]
    null = this_df['null'].values[0]
    
    for var in var_names:
        s = this_df[var].values[0]
        alpha_grid = np.linspace(alpha_interval[0],alpha_interval[1],s.shape[0])

        fig, ax = plt.subplots(1,1,figsize=(10,8))
        img = ax.imshow(s[0,:,:], interpolation='nearest', cmap='binary', vmin=-1, vmax=1)
        cbar = fig.colorbar(img)
        ttl = plt.text(0, 1.01, '', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, bitrate=3600)
        
        ani = animation.FuncAnimation(fig, animate, frames=s.shape[0])
        ani.save(os.path.join(vid_dir,run_id+'_'+var+'.mp4'), writer=writer)