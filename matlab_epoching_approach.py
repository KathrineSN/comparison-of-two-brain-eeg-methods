# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:21:36 2021

@author: kathr
"""
# Importing dependencies
import os
path="C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Connectivity-Special-Course-2"
os.chdir(path)
import mne
import numpy as np
import matplotlib.pyplot as plt
from con_measures import *
from plot_functions import *


#%% Loading the complex signal of coupled from the 25 seconds epochs

def loading_long_epochs(pair_no):
    epochs_a_long = mne.read_epochs('Preprocessed data\\subj_a_long_' + str(pair_no) +'.fif')
    epochs_b_long = mne.read_epochs('Preprocessed data\\subj_b_long_' + str(pair_no) +'.fif')
        
    alpha, result, filtered, complex_signal, epo_a_c = plv(epochs_a_long, epochs_b_long, 'pair00' + str(pair_no), '25sec', save = False, plot = False)
    
    segments = []

    for i in range(16):
        start_j = 0
        for j in range(1,8):
            print(j)
            print(start_j)
            print(start_j + (256*3))
            segments.append(complex_signal[:,i,:,:,start_j:start_j + (256*3)])                     
            start_j += (256*3)
    
    return segments

#%% Average connectivity for each epoch

def manual_plv(x1,x2):
    phase_diff = np.angle(x1)- np.angle(x2)
    con = abs(np.mean(np.exp(1j*phase_diff)))
    
    return con

def plv_per_trial(trial_no, segments):
    matrices = []
    con_vals = np.zeros((64,64))
    trial_no = trial_no
    
    for h in range(7):
        print(trial_no)
        print(h)
        for i in range(64):
            for j in range(64):
                
                con_vals[i,j] = manual_plv(segments[h+(trial_no*7)][0,i,1,:], segments[h+(trial_no*7)][1,j,1,:])
            
        matrices.append(con_vals)
        con_vals = np.zeros((64,64))
    
    avg_mat = sum(np.array(matrices))/7
    
    from plot_functions import heatmap
    
    fig, ax = plt.subplots(figsize=(16,16))
    plt.suptitle('Definition from Li: ' + str(trial_no) ,fontsize = 25)
    #heatmap = ax.pcolor(coupled, cmap=plt.cm.Reds, alpha=0.8)
    montage = mne.channels.make_standard_montage("biosemi64")
    new_ch_names = montage.ch_names
    
    im, cbar = heatmap(avg_mat, new_ch_names, new_ch_names, ax=ax,
                       cmap=plt.cm.Reds, cbarlabel='plv')
    
    return avg_mat

#%% Average over epochs related to coupled

def compute_pair_matrix(pair_no, segments):
    average_matrices = []
    
    for i in range(16):
        #print(i)
        avg_mat = plv_per_trial(i, segments)
        average_matrices.append(avg_mat)
    
    print(len(average_matrices))
    coupled_mat = sum(np.array(average_matrices))/16
        
    fig, ax = plt.subplots(figsize=(16,16))
    plt.suptitle('PLV Coupled Pair' + str(pair_no) ,fontsize = 25)
    #heatmap = ax.pcolor(coupled, cmap=plt.cm.Reds, alpha=0.8)
    montage = mne.channels.make_standard_montage("biosemi64")
    new_ch_names = montage.ch_names
    
    im, cbar = heatmap(coupled_mat, new_ch_names, new_ch_names, ax=ax,
                       cmap=plt.cm.Reds, cbarlabel='plv')
    plt.savefig('Connectivity matrices\\plv\\images\\average pair matrices segmented 25s - 02-12-21\\plv_pair00' + str(pair_no) + '_alpha_Coupled_3sec(25).png')
    np.save('Connectivity matrices\\plv\\plv per pair segmented 25s epochs - 02-12-21\\plv_pair00' + str(pair_no) + '_alpha_Coupled_3sec(25).npy', coupled_mat)
    
    return coupled_mat

#%%
subj_no = np.array([3,4,5,7,9,10,11,12,14,16,17,18,19,20,22,23,24,27])
for i in subj_no:
    segments = loading_long_epochs(i)
    coupled_mat = compute_pair_matrix(i, segments)