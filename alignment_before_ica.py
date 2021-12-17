# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:05:18 2021

@author: kathr
"""

# Importing dependencies
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Setting path
path="C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Connectivity-Special-Course-2"
os.chdir(path)

#%% Loading epochs before ICA

epochs_a = mne.read_epochs('Epochs before ICA\\subja_3sec_4.fif')
epochs_b = mne.read_epochs('Epochs before ICA\\subjb_3sec_4.fif')

#%% Loading mat-files before ICA

# Alignment in last coupled trial
folder ='C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Connectivity-Special-Course-2\\Mat-files before ICA\\'
last_coupled = scipy.io.loadmat(folder+'eeg_coupled_last_2.mat')

keys_1 = []
for key_1 in last_coupled.keys():
    keys_1.append(key_1)
matrix = last_coupled[keys_1[-1]]

df = epochs_a._data[556][12][:]

plt.figure(figsize = (12,6))
t = np.arange(20,23,3/768)
plt.plot(t,(matrix[12][5888:6656]-np.mean(matrix[12][5888:6656]))/np.std(matrix[12][5888:6656]), label = 'Marius')
plt.plot(t,(df-np.mean(df))/np.std(df), label = 'Kathrine')
#plt.plot(t,matrix[12][1280:2048], label = 'Marius')
#plt.plot(t,df, label = 'Kathrine')
plt.title('Last Coupled trial for C3')
plt.legend()
plt.xlabel('time (s)')

#%% Alignment in first coupled trial
# 3 seconds from first coupled trial for channel C3
first_coupled = scipy.io.loadmat(folder+'eeg_first_coupled.mat')

keys_1 = []
for key_1 in first_coupled.keys():
    keys_1.append(key_1)
matrix_c = first_coupled[keys_1[-1]]

df_c = epochs_a._data[10][12][:]

#%% Loading final received data from Marius
folder ='C:/Users/kathr/OneDrive/Documents/GitHub/Connectivity-Special-Course-2/Final mat-file/'
subj1_final = scipy.io.loadmat(folder+'eeg_pair004.mat')

# Creating list of numpy arrays from matlab file
# Subject 1
keys_1_final = []
for key in subj1_final.keys():
    keys_1_final.append(key)
subj_1_final = subj1_final[keys_1_final[-1]]
#%%

plt.figure(figsize = (8,2.5))
t = np.arange(2,5,3/768)
#plt.plot(t,(matrix_c[12][1280:2048]-np.mean(matrix_c[12][1280:2048]))/np.std(matrix_c[12][1280:2048]), label = 'Fieldtrip')
plt.plot(t,(df_c-np.mean(df_c))/np.std(df_c), label = 'HyPyP')
plt.plot(t,(subj_1_final[0,0:768]-np.mean(subj_1_final[0,0:768]))/np.std(subj_1_final[0,0:768]), label = 'Fieldtrip')
plt.title('C3 signal before ICA', fontsize = 12)
plt.legend(fontsize = 9)
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('signal (' + u'\u03bc'+'V)', fontsize = 9)
plt.tight_layout()
plt.savefig("Figures for report\\before ICA.png",bbox_inches='tight',dpi=300)








