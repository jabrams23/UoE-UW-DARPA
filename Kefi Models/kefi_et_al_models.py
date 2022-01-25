# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 12:47:57 2021

@author: Daniel Dylewsky

Implementation of models from Kefi et. al., "Early Warning Signals of
Ecological Transitions: Methods for Spatial Patterns" (2014)

"""

import numpy as np
import matplotlib.pyplot as plt
import os

def sim_model(fn,x0,run_time,burn_time,consts,alpha_interval,dt=1,save_every=1,return_burn=False,pbc=True):
    # Burn in and simulate the system dx/dt = fn(x) starting from x0
    #
    # ALL DYNAMICAL VARIABLES ARE ASSUMED TO BE POSITIVE DEFINITE AND FORCIBLY BOUNDED AS SUCH
    # This is consistent with the systems being modeled in systems 1 and 3 from Kefi et. al.
    
    burn_steps = int(burn_time/dt)
    burn_steps_save = int((burn_time/dt)/save_every)
    run_steps = int(run_time/dt)
    run_steps_save = int((run_time/dt)/save_every)
    x_burn = np.zeros((burn_steps_save,len(x0)))
    x = np.zeros((run_steps_save,len(x0)))
    alpha_burn = alpha_interval[0]
    print('Running burn')
    this_x_burn = x0
    k = 0
    for j in range(burn_steps):
        this_x_burn = this_x_burn + dt*fn(this_x_burn,consts,alpha_burn,pbc=pbc)
        this_x_burn[this_x_burn<0] = 0
        if j % save_every == save_every-1:
            x_burn[k,:] = this_x_burn
            k += 1
    
    alpha = np.linspace(alpha_interval[0],alpha_interval[1],run_steps)
    print('Running sim')
    k = 0
    this_x = x_burn[-1,:]
    for j in range(run_steps):
        this_x = this_x + dt*fn(this_x,consts,alpha[j],pbc=pbc)
        this_x[this_x<0] = 0
        if j % save_every == save_every-1:
            x[k,:] = this_x
            k += 1
    
    if return_burn:
        return (x, x_burn)
    else:
        return x
    
out_dir = 'Kefi_Models'

    
# %% Model 1: Local positive feedback model with no patchy pattern

def dw(w,B,consts,alpha,pbc=True):
    if pbc:
        w_pad = w
        B_pad = B
    else:
        w_pad = np.zeros((w.shape[0]+2,w.shape[1]+2))
        w_pad[1:-1,1:-1] = w
        B_pad = np.zeros((B.shape[0]+2,B.shape[1]+2))
        B_pad[1:-1,1:-1] = B
        
    dw = alpha - w_pad - consts['lamb']*np.multiply(w_pad,B_pad) + \
        consts['D']*(np.roll(w_pad,(1,0),axis=(0,1))+np.roll(w_pad,(-1,0),axis=(0,1))+np.roll(w_pad,(0,1),axis=(0,1))+np.roll(w_pad,(0,-1),axis=(0,1))-4*w_pad) + \
        consts['sigma_w']*np.random.randn(*w_pad.shape)
        
    if not pbc:
        dw = dw[1:-1,1:-1]
    return dw

def dB(w,B,consts,pbc=True):
    if pbc:
        w_pad = w
        B_pad = B
    else:
        w_pad = np.zeros((w.shape[0]+2,w.shape[1]+2))
        w_pad[1:-1,1:-1] = w
        B_pad = np.zeros((B.shape[0]+2,B.shape[1]+2))
        B_pad[1:-1,1:-1] = B
        
    dB = consts['rho']*np.multiply(B_pad,(w_pad-(1/consts['B_c'])*B_pad)) - \
        consts['mu']*np.divide(B_pad,(B_pad+consts['B_0'])) + \
        consts['D']*(np.roll(B_pad,(1,0),axis=(0,1))+np.roll(B_pad,(-1,0),axis=(0,1))+np.roll(B_pad,(0,1),axis=(0,1))+np.roll(B_pad,(0,-1),axis=(0,1))-4*B_pad) + \
        consts['sigma_B']*np.random.randn(*B_pad.shape)
        
    if not pbc:
        dB = dB[1:-1,1:-1]
    return dB


def x_to_wB(x):
    if len(x.shape)==1:
        s2 = int(x.shape[0]/2)
        s = int(np.sqrt(s2))
        w = x[:s2].reshape((s,s))
        B = x[s2:].reshape((s,s))
    elif len(x.shape)==2:
        s2 = int(x.shape[1]/2)
        s = int(np.sqrt(s2))
        w = x[:,:s2]
        w = w.reshape(w.shape[0],s,s)
        B = x[:,s2:]
        B = B.reshape(B.shape[0],s,s)
    return(w,B)

def dx_m1(x,consts,alpha,pbc=True):
    w, B = x_to_wB(x)
    
    this_dw = dw(w,B,consts,alpha,pbc=pbc)
    this_dB = dB(w,B,consts,pbc=pbc)
    dx = np.hstack((this_dw.reshape(-1),this_dB.reshape(-1)))
    return dx



def generate_model1(consts,alpha_interval,rj,lattice_size=200,delta_t=0.1,save_every=10,burn_time=500,run_time=2000,plot_res=False,return_burn=False):
    
    
    w0 = (1/lattice_size)*np.random.uniform(size=(lattice_size,lattice_size))
    B0 = (1/lattice_size)*np.random.uniform(size=(lattice_size,lattice_size))
    x0 = np.hstack((w0.reshape(-1),B0.reshape(-1)))
    
    if return_burn:
        x, x_burn = sim_model(dx_m1,x0,run_time,burn_time,consts,alpha_interval,dt=delta_t,save_every = save_every,return_burn=return_burn,pbc=True)
        w_burn, B_burn = x_to_wB(x_burn)
    else:
        x = sim_model(dx_m1,x0,run_time,burn_time,consts,alpha_interval,dt=delta_t,save_every = save_every,return_burn=return_burn,pbc=True)
    w,B = x_to_wB(x)
    
    
    if plot_res:
        for vj,vn in enumerate(['w','B']):
            if vj == 0:
                this_qoi = w
            elif vj == 1:
                this_qoi = B
            fig, axs = plt.subplots(1,1,figsize=(8,6))
            axs.plot(np.mean(this_qoi,axis=(1,2)))
            axs.set_title(vn)
    
            if not os.path.exists(os.path.join('Kefi_Models','Plots','Model 1',vn)):
                os.makedirs(os.path.join('Kefi_Models','Plots','Model 1',vn))
                
            plt.savefig(os.path.join('Kefi_Models','Plots','Model 1',vn,'Model_1_Output_{:03d}.png'.format(rj)))
            plt.close()
            
            if return_burn:
                if vj == 0:
                    this_qoi = w_burn
                elif vj == 1:
                    this_qoi = B_burn
                fig, axs = plt.subplots(1,1,figsize=(8,6))
                axs.plot(np.mean(this_qoi,axis=(1,2)))
                axs.set_title(vn+' Burn')
                if not os.path.exists(os.path.join('Kefi_Models','Plots','Model 1',vn+'_burn')):
                    os.makedirs(os.path.join('Kefi_Models','Plots','Model 1',vn+'_burn'))
                plt.savefig(os.path.join('Kefi_Models','Plots','Model 1',vn+'_burn','Model_1_Output_{:03d}.png'.format(rj)))
                plt.close()
    
    if plot_res:
        fig, axs = plt.subplots(2,2,figsize=(8,6))
        axs[0,0].plot(np.mean(w_burn,axis=(1,2)))
        axs[0,0].set_title('w Burn')
        axs[1,0].plot(np.mean(B_burn,axis=(1,2)))
        axs[1,0].set_title('B Burn')
        axs[0,1].plot(np.mean(w,axis=(1,2)))
        axs[0,1].set_title('w')
        axs[1,1].plot(np.mean(B,axis=(1,2)))
        axs[1,1].set_title('B')
        # axs[0,1].plot(alpha_grid,np.mean(w,axis=(1,2)))
        # axs[0,1].set_title('w')
        # axs[1,1].plot(alpha_grid,np.mean(B,axis=(1,2)))
        # axs[1,1].set_title('B')
        plt.show()
        
        w_vlim = [np.min(w),np.max(w)]
        B_vlim = [np.min(B),np.max(B)]
        fig, axs = plt.subplots(2,2,figsize=(8,6))
        axs[0,0].imshow(w[0,:,:],vmin=w_vlim[0],vmax=w_vlim[1])
        axs[0,0].set_title('w[0]')
        axs[1,0].imshow(B[0,:,:],vmin=B_vlim[0],vmax=B_vlim[1])
        axs[1,0].set_title('B[0]')
        axs[0,1].imshow(w[-1,:,:],vmin=w_vlim[0],vmax=w_vlim[1])
        axs[0,1].set_title('w[-1]')
        axs[1,1].imshow(B[-1,:,:],vmin=B_vlim[0],vmax=B_vlim[1])
        axs[1,1].set_title('B[-1]')
        plt.show()
    
    out_dict = {'w':[w], 'B':[B], 'alpha_interval':[alpha_interval]}
    if return_burn:
        out_dict['w_burn'] = [w_burn]
        out_dict['B_burn'] = [B_burn]
        
    return out_dict


# %% Model 2: Local facilitation model, yielding scale-free patchy vegetation

def q(w,v1,v2):
    # probability that a neighbor of a site in state v1 has value v2
    w1 = (w==v1)
    w2 = (w==v2)
    z = np.ones(w.shape)
    shift_list = [(1,0),(-1,0),(0,1),(0,-1)]
    pN = 0
    pD = 0
    for shift in shift_list:
        pN += np.sum(np.multiply(np.roll(w1,shift,axis=(0,1)),w2))
        pD += np.sum(np.multiply(np.roll(w1,shift,axis=(0,1)),z))
    if pD == 0:
        # if there are no sites in state v1
        return 0
    else:
        return pN/pD
    
def w_next(w,consts,alpha,i,j):
    wij = w[i,j]
    if wij == 1:
        p = [0,consts['m'],1-consts['m']] # -1, 0, 1
    elif wij == -1:
        tp = consts['r']+consts['f']*q(w,1,-1)
        p = [1-tp, tp, 0]
    else:
        rho_plus = np.count_nonzero(w==1)/np.size(w)
        p_plus = (consts['delta']*rho_plus + (1-consts['delta'])*q(w,1,0))*(alpha-consts['c']*rho_plus)
        p_minus = consts['d']
        p_naught = 1-(p_plus+p_minus)
        p = [p_minus,p_naught,p_plus]
    if np.any(np.isnan(p)):
        print('Probabilities contain NaN')
        import pdb; pdb.set_trace()
    if np.any(np.array(p)<0):
        print('Negative-valued probability')
        import pdb; pdb.set_trace()
    w[i,j] = np.random.choice([-1,0,1],p=p)
    return w

def generate_model2(consts,alpha_interval,rj,lattice_size=64,delta_t=1,save_every=1,burn_time=400,run_time=800,plot_res=False,return_burn=False):
    # discrete model, time is just measured in steps:
    burn_steps = burn_time
    run_steps = run_time
    
    generation_length = int(lattice_size**2)
    
    
    # desert equilibrium as defined in paper:
    p0 = [consts['d']/(consts['d']+consts['r']),consts['r']/(consts['d']+consts['r']),0]
    # desert stability condition:
    if alpha_interval[0]<consts['m']*(consts['d']+consts['r'])/consts['r']:
        print('Desert equilibrium IC is stable')
    else:
        print('Desert equilibrium IC is unstable')
        import pdb; pdb.set_trace()
    
    w0 = np.random.choice([-1,0,1],size=(lattice_size,lattice_size), p=p0)
    
    w_burn = np.zeros((burn_steps,lattice_size,lattice_size))
    w = np.zeros((run_steps,lattice_size,lattice_size))
    
    alpha_burn = alpha_interval[0]
    alpha_grid = np.linspace(alpha_interval[0],alpha_interval[1],run_steps)
    
    w_burn[0,:,:] = w0
    
    for jj in range(1,burn_steps):
        if jj%10 == 0:
            print('Burn generation ' + str(jj) + '/' + str(burn_steps))
        this_w = w_burn[jj-1,:,:]
        this_w_avg = np.zeros(this_w.shape)
        for kk in range(generation_length):
            site_i = np.random.randint(lattice_size)
            site_j = np.random.randint(lattice_size)
            this_w = w_next(this_w,consts,alpha_burn,site_i,site_j)
            this_w_avg = this_w_avg + this_w
        this_w_avg = this_w_avg/generation_length
        w_burn[jj,:,:] = this_w_avg
        
    w[0,:,:] = w_burn[-1,:,:]
    for jj in range(1,run_steps):
        if jj%10 == 0:
            print('Run generation ' + str(jj) + '/' + str(run_steps))
        this_w = w[jj-1,:,:]
        this_w_avg = np.zeros(this_w.shape)
        for kk in range(generation_length):
            site_i = np.random.randint(lattice_size)
            site_j = np.random.randint(lattice_size)
            this_w = w_next(this_w,consts,alpha_grid[jj],site_i,site_j)
            this_w_avg = this_w_avg + this_w
        this_w_avg = this_w_avg/generation_length
        w[jj,:,:] = this_w_avg
        
    if plot_res:
        fig, axs = plt.subplots(1,1,figsize=(8,6))
        axs.plot(np.mean(w,axis=(1,2)))
        axs.set_title('w')

        
        # w_vlim = [np.min(w),np.max(w)]
        # fig, axs = plt.subplots(1,2,figsize=(8,6))
        # axs[0].imshow(w[0,:,:],vmin=w_vlim[0],vmax=w_vlim[1])
        # axs[0].set_title('w[0]')
        # axs[1].imshow(w[-1,:,:],vmin=w_vlim[0],vmax=w_vlim[1])
        # axs[1].set_title('w[-1]')
        
        if not os.path.exists(os.path.join('Kefi_Models','Plots','Model 2','w')):
            os.makedirs(os.path.join('Kefi_Models','Plots','Model 2','w'))
            
        plt.savefig(os.path.join('Kefi_Models','Plots','Model 2','w','Model_2_Output_{:03d}.png'.format(rj)))
        plt.close()
        
        if return_burn:
            fig, axs = plt.subplots(1,1,figsize=(8,6))
            axs.plot(np.mean(w_burn,axis=(1,2)))
            axs.set_title('w Burn')
            if not os.path.exists(os.path.join('Kefi_Models','Plots','Model 2','w_burn')):
                os.makedirs(os.path.join('Kefi_Models','Plots','Model 2','w_burn'))
            plt.savefig(os.path.join('Kefi_Models','Plots','Model 2','w_burn','Model_2_Output_{:03d}.png'.format(rj)))
            plt.close()

    out_dict = {'w':[w], 'alpha_interval':[alpha_interval]}
    if return_burn:
        out_dict['w_burn'] = [w_burn]
        
    return out_dict
# %% Model 3: Scale-dependent feedback model, yielding periodic patterns

alpha_crit = 0.875 # estimate for these parameters
# alpha_interval = [2,0.05]
alpha_interval = [2,0.5]

def del2(X):
    del2x = np.roll(X,(1,0),axis=(0,1))+np.roll(X,(-1,0),axis=(0,1))+np.roll(X,(0,1),axis=(0,1))+np.roll(X,(0,-1),axis=(0,1))-4*X
    return del2x

def dO(O,W,P,consts,alpha):
    dO = alpha - consts['a']*np.multiply(O,np.divide(P+consts['W_0']*consts['k_2'],P+consts['k_2'])) + \
        consts['D_o']*del2(O) + consts['sigma']*np.random.randn(*O.shape)
    return dO

def dW(O,W,P,consts,alpha):
    dW = consts['a']*np.multiply(O,np.divide(P+consts['W_0']*consts['k_2'],P+consts['k_2'])) - \
        consts['g_max']*np.multiply(np.divide(W,W+consts['k_1']),P) - \
        consts['r_w']*W + consts['D_w']*del2(W) + consts['sigma']*np.random.randn(*W.shape)
    return dW

def dP(O,W,P,consts,alpha):
    dP = np.multiply(consts['c']*consts['g_max']*np.divide(W,W+consts['k_1'])-consts['d'],P) + \
        consts['D_p']*del2(P) + consts['sigma']*np.random.randn(*P.shape)
    # dP2 = consts['c']*consts['g_max']*np.multiply(np.divide(W,W+consts['k_1']),P) - \
    #     consts['d']*P + consts['D_p']*del2(P)
    
    return dP



def x_to_OWP(x):
    if len(x.shape)==1:
        s2 = int(x.shape[0]/3)
        s = int(np.sqrt(s2))
        O = x[:s2].reshape((s,s))
        W = x[s2:(2*s2)].reshape((s,s))
        P = x[(2*s2):].reshape((s,s))
    elif len(x.shape)==2:
        s2 = int(x.shape[1]/3)
        s = int(np.sqrt(s2))
        O = x[:,:s2].reshape((x.shape[0],s,s))
        W = x[:,s2:(2*s2)].reshape((x.shape[0],s,s))
        P = x[:,(2*s2):].reshape((x.shape[0],s,s))
    return(O,W,P)

def dx_m3(x,consts,alpha,pbc=True):
    O,W,P = x_to_OWP(x)
    
    this_dO = dO(O,W,P,consts,alpha)
    this_dW = dW(O,W,P,consts,alpha)
    this_dP = dP(O,W,P,consts,alpha)
    dx = np.hstack((this_dO.reshape(-1),this_dW.reshape(-1),this_dP.reshape(-1)))
    if np.any(O < 0):
        print('O negative') 
        import pdb; pdb.set_trace()
    if np.any(W < 0):
        print('W negative') 
        import pdb; pdb.set_trace()
    if np.any(P < 0):
        print('P negative') 
        import pdb; pdb.set_trace()
    return dx

def generate_model3(consts,alpha_interval,rj,lattice_size=128,delta_t=0.002,save_every=400,burn_time=200,run_time=400,plot_res=False,return_burn=False):
    O_0 = np.ones((lattice_size,lattice_size))+0.001*np.random.randn(lattice_size,lattice_size)
    W_0 = np.ones((lattice_size,lattice_size))+0.001*np.random.randn(lattice_size,lattice_size)
    P_0 = 5+0.001*np.random.randn(lattice_size,lattice_size)
    
    # O_0 = (1/lattice_size)*np.random.uniform(size=(lattice_size,lattice_size))
    # W_0 = (1/lattice_size)*np.random.uniform(size=(lattice_size,lattice_size))
    # P_0 = (1/lattice_size)*np.random.uniform(size=(lattice_size,lattice_size))
    x0 = np.hstack((O_0.reshape(-1),W_0.reshape(-1),P_0.reshape(-1)))
    
    x, x_burn = sim_model(dx_m3,x0,run_time,burn_time,consts,alpha_interval,dt=delta_t,save_every=save_every,return_burn=return_burn)
    O,W,P = x_to_OWP(x)
    O_burn,W_burn,P_burn = x_to_OWP(x_burn)
    
    
    if plot_res:
        for vj,vn in enumerate(['O','W','P']):
            if vj == 0:
                this_qoi = O
            elif vj == 1:
                this_qoi = W
            elif vj == 2:
                this_qoi = P
            fig, axs = plt.subplots(1,1,figsize=(8,6))
            axs.plot(np.mean(this_qoi,axis=(1,2)))
            axs.set_title(vn)
    
            if not os.path.exists(os.path.join('Kefi_Models','Plots','Model 3',vn)):
                os.makedirs(os.path.join('Kefi_Models','Plots','Model 3',vn))
                
            plt.savefig(os.path.join('Kefi_Models','Plots','Model 3',vn,'Model_3_Output_{:03d}.png'.format(rj)))
            plt.close()
            
            if return_burn:
                if vj == 0:
                    this_qoi = O_burn
                elif vj == 1:
                    this_qoi = W_burn
                elif vj == 2:
                    this_qoi = P_burn
                fig, axs = plt.subplots(1,1,figsize=(8,6))
                axs.plot(np.mean(this_qoi,axis=(1,2)))
                axs.set_title(vn+' Burn')
                if not os.path.exists(os.path.join('Kefi_Models','Plots','Model 3',vn+'_burn')):
                    os.makedirs(os.path.join('Kefi_Models','Plots','Model 3',vn+'_burn'))
                plt.savefig(os.path.join('Kefi_Models','Plots','Model 3',vn+'_burn','Model_3_Output_{:03d}.png'.format(rj)))
                plt.close()

    out_dict = {'O':[O], 'W':[W], 'P':[P], 'alpha_interval':[alpha_interval]}
    if return_burn:
        out_dict['O_burn'] = [O_burn]
        out_dict['W_burn'] = [W_burn]
        out_dict['P_burn'] = [P_burn]
        
    return out_dict