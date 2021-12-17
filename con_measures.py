# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 15:04:27 2021

@author: kathr
"""

# Importing dependencies
import os
import mne
import numpy as np
from hypyp import prep 
from hypyp import analyses
from collections import OrderedDict
from itertools import groupby
from plot_functions import heatmap
import matplotlib.pyplot as plt

# Setting path
path="C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Connectivity-Special-Course-2"
os.chdir(path)

# Function for Phase Locking Value
def plv(epochs_a, epochs_b, pair_name, length, save = False, plot = False):
    
    
    event_dict = {'Uncoupled': 102, 'Coupled': 103, 'Leader': 105,
              'Follower': 107, 'Control':108 }

    conditions = ['Coupled']

    if length == '25sec':
        epo_a_cleaned = epochs_a.crop(tmin = 2, tmax = 23)
        #epochs_a.plot(n_epochs = 1, n_channels = 10)
        epo_b_cleaned = epochs_b.crop(tmin = 2, tmax = 23)
    
    if length == '3sec':
        epo_drop = []
        epo_drop.append(0)
        epo_drop.append(8)
        for i in range(63*2): #Previously was 64*5 changed as first trial is already appended
            epo_drop.append(epo_drop[i]+9)
        print(len(epo_drop))
        
        # Dropping the beginning and end of a trial      
        epo_a = epochs_a.drop(epo_drop)        
        epo_b = epochs_b.drop(epo_drop)
        
        epo_a_copy = epo_a.copy()
        epo_b_copy = epo_b.copy()
    
        # Running autoreject function
        cleaned_epochs_AR, dic_AR = prep.AR_local([epo_a_copy, epo_b_copy],
                                        strategy="union",
                                        threshold=50.0,
                                        verbose=True)
        
        epo_a_cleaned = cleaned_epochs_AR[0]
        epo_b_cleaned = cleaned_epochs_AR[1]
    
    if length == '1sec':
        epo_drop = []
        epo_drop.append(0)
        epo_drop.append(1)
        epo_drop.append(2)
        epo_drop.append(24)
        epo_drop.append(25)
        for i in range(63*5): #Previously was 64*5 changed as first trial is already appended
            epo_drop.append(epo_drop[i]+26)
        print(len(epo_drop))
        
        # Dropping the beginning and end of a trial      
        epo_a = epochs_a.drop(epo_drop)        
        epo_b = epochs_b.drop(epo_drop)
        
        epo_a_copy = epo_a.copy()
        epo_b_copy = epo_b.copy()
        
        cleaned_epochs_AR, dic_AR = prep.AR_local([epo_a_copy, epo_b_copy],
                                        strategy="union",
                                        threshold=50.0,
                                        verbose=True)
        
        epo_a_cleaned = cleaned_epochs_AR[0]
        epo_b_cleaned = cleaned_epochs_AR[1]
    
    # Getting the number of epochs of specific condition in a row
    a = epo_a_cleaned.events[:,2]
    d = dict()
    
    for k, v in groupby(a):
        d.setdefault(k, []).append(len(list(v)))
    print(d)
    
    for c in conditions:
        epo_a_c = epo_a_cleaned[c]
        epo_b_c = epo_b_cleaned[c]
        
        freq_bands = {'Theta': [4, 7],
                    'Alpha' :[8, 13],
                    'Beta': [15, 25]}
        
        sampling_rate = epo_a_c.info['sfreq']
                            
        #Connectivity
        
        #Data and storage
        data_inter = np.array([epo_a_c, epo_b_c])
        
        #Analytic signal per frequency band
        complex_signal, filtered = analyses.compute_freq_bands(data_inter, sampling_rate,
                                                     freq_bands)
        
        #Getting connectivity values
        result,_,_,_ = analyses.compute_sync(complex_signal, mode='plv', epochs_average = False)
        
        #Defining the number of channels
        n_ch = len(epochs_a.info['ch_names'])
        
        #Averaging over the epochs specific to the given trial
        
        result_theta = result[0]
        result_alpha = result[1]
        result_beta = result[2]
        
        results = [result_theta, result_alpha, result_beta]
        bands = ['theta','alpha','beta']
        
        for h in range(len(bands)):
            print(h)
            start_i = 0
            epoch_no = 1
            epoch_matrices = []
            for i in d[event_dict[c]]:
                print(i)
                band_mat = results[h]
                print(band_mat.shape)
                mat = sum(band_mat[start_i:(start_i+i),0:n_ch,n_ch:2*n_ch])/i
                print(mat)
                epoch_matrices.append(mat)
                if save:
                    np.save('Connectivity matrices/plv/plv per trial/' + 'plv_' + pair_name + '_' + bands[h] + '_' + c + '_' + length + '_' + str(epoch_no), mat)
                start_i += i
                epoch_no += 1
            
            # Mean of all trials related to condition 'c'
            print(len(epoch_matrices))
            mean_matrix = np.mean(epoch_matrices, axis = 0)
            print(mean_matrix.shape)
            if save: 
                np.save('Connectivity matrices/plv/plv per pair/' + 'plv_' + pair_name + '_' + bands[h] + '_' + c + '_' + length, mean_matrix)
            
            if plot:           
                fig, ax = plt.subplots(figsize=(16,16))
                plt.suptitle('plv ' + bands[h] + ' ' + pair_name + ' ' + c,fontsize = 25)
                montage = mne.channels.make_standard_montage("biosemi64")
                new_ch_names = montage.ch_names
                im, cbar = heatmap(mean_matrix, new_ch_names, new_ch_names, ax=ax,
                               cmap=plt.cm.Reds, cbarlabel='plv')
                plt.show()
                
                savepath = 'C:/Users/kathr/OneDrive/Documents/GitHub/Connectivity-Special-Course-2/Connectivity matrices/plv/images/pair matrices - 28-11-21'
                plt.savefig(savepath + '/plv_' + c + '_' + bands[h] + '_' + pair_name + '_' + length + '.png')
                
    return result_alpha, result, filtered, complex_signal, epo_a_c

