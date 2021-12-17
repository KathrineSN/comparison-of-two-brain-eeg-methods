# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 15:20:27 2021

@author: kathr
"""
# Importing dependencies
import os
path="C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Connectivity-Special-Course-2"
os.chdir(path)
import mne
import numpy as np
from con_measures import *
from plot_functions import *

# Loading epochs from a pair
epochs_a_4 = mne.read_epochs('Preprocessed data\\subj_a_3sec_4.fif')
epochs_b_4 = mne.read_epochs('Preprocessed data\\subj_b_3sec_4.fif')

#%%
# Calculating plv
alpha_plv, result_plv, filtered_plv2, complex_signal_plv2, epo_a_c = plv(epochs_a_4, epochs_b_4, 'pair004', '3sec', save = True, plot = True)

#%% Looping over the 18 subjects

subj_no = np.array([3,4,5,7,9,10,11,12,14,16,17,18,19,20,22,23,24,27])

#for i in range(len(subj_no)):
i = 16
    
# 3 seconds epochs
epochs_a_3sec = mne.read_epochs('Preprocessed data\\subj_a_3sec_' + str(subj_no[i]) + '.fif')
epochs_b_3sec = mne.read_epochs('Preprocessed data\\subj_b_3sec_' + str(subj_no[i]) + '.fif')

plv(epochs_a_3sec, epochs_b_3sec, 'pair00' + str(subj_no[i]), '3sec', save = True, plot = True)

# 1 seconds epochs
epochs_a_short = mne.read_epochs('Preprocessed data\\subj_a_short_' + str(subj_no[i]) + '.fif')
epochs_b_short = mne.read_epochs('Preprocessed data\\subj_b_short_' + str(subj_no[i]) + '.fif')

plv(epochs_a_short, epochs_b_short, 'pair00' + str(subj_no[i]), '1sec', save = True, plot = True)

# 25 seconds epochs
epochs_a_long = mne.read_epochs('Preprocessed data\\subj_a_long_' + str(subj_no[i]) + '.fif')
epochs_b_long = mne.read_epochs('Preprocessed data\\subj_b_long_' + str(subj_no[i]) + '.fif')

plv(epochs_a_long, epochs_b_long, 'pair00' + str(subj_no[i]), '25sec', save = True, plot = True)