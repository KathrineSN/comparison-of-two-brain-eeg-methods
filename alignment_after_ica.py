# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 09:49:31 2021

@author: kathr
"""

# Import dependecies
import os
path="C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Connectivity-Special-Course-2\\"
os.chdir(path)
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from hypyp import analyses
import mne

#%% Loading preprocessed (after ICA, before filtering and re-ferencing)
epochs_a_4 = mne.read_epochs('Epochs without re-referencing\\subj_a_3sec_4_noref.fif')
epochs_b_4 = mne.read_epochs('Epochs without re-referencing\\subj_b_3sec_4_noref.fif')

epo_pre = epochs_a_4['Coupled']._data

first_epoch = []
for i in range(7):
    i += 1
    print(i)
    first_epoch.append(epo_pre[i,12,:])
  
pre_to_plot = np.hstack(first_epoch)    
  
#%% Loading corresponding data from Marius
folder ='C:/Users/kathr/OneDrive/Documents/GitHub/Connectivity-Special-Course-2/5 coupled epochs after ICA from Marius/'
subj1_m = scipy.io.loadmat(folder+'prefilter_ppn1.mat')
subj2_m = scipy.io.loadmat(folder+'prefilter_ppn2.mat')

# Creating list of numpy arrays from matlab file
# Subject 1
keys_1 = []
for key_1 in subj1_m.keys():
    keys_1.append(key_1)
subj_1 = subj1_m[keys_1[-1]]

# Subject 2
keys_2 = []
for key_2 in subj2_m.keys():
    keys_2.append(key_2)
subj_2 = subj2_m[keys_2[-1]]

#%% Loading final received data from Marius
folder ='C:/Users/kathr/OneDrive/Documents/GitHub/Connectivity-Special-Course-2/Final mat-file/'
subj1_final = scipy.io.loadmat(folder+'eeg_pair004.mat')

# Creating list of numpy arrays from matlab file
# Subject 1
keys_1_final = []
for key in subj1_final.keys():
    keys_1_final.append(key)
subj_1_final = subj1_final[keys_1_final[-1]]



#%% Comparison of the first coupled epoch
# the first coupled epoch
plt.figure()
plt.title('The first coupled trial')
plt.plot((subj_1[0,12,:]-np.mean(subj_1[0,12,:]))/np.std(subj_1[0,12,:]), label = 'Marius')
plt.plot((pre_to_plot[0:5377]-np.mean(pre_to_plot[0:5377]))/np.std(pre_to_plot[0:5377]),label = 'Kathrine')
plt.legend()

#%% the first 3 seconds of the first coupled epoch
t = np.arange(2,5,3/768)
plt.figure(figsize = (8,2.5))
plt.title('C3 signal after ICA', fontsize = 12)
#plt.plot(t,(subj_1[0,12,0:768]-np.mean(subj_1[0,12,0:768]))/np.std(subj_1[0,12,0:768]), label = 'Fieldtrip')
plt.plot(t,(pre_to_plot[0:768]-np.mean(pre_to_plot[0:768]))/np.std(pre_to_plot[0:768]),label = 'HyPyP')
plt.plot(t,(subj_1_final[1,0:768]-np.mean(subj_1_final[1,0:768]))/np.std(subj_1_final[1,0:768]), label = 'Fieldtrip')
plt.legend(fontsize = 9)
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('signal (' + u'\u03bc'+'V)', fontsize = 9)
plt.tight_layout()
plt.savefig("Figures for report\\after ICA.png",bbox_inches='tight',dpi=1200)

#%% Comparison of first coupled epoch before and after ICA HyPyP
epochs_a_preICA = mne.read_epochs('Epochs before ICA\\subja_3sec_4.fif')
epochs_b_preICA = mne.read_epochs('Epochs before ICA\\subjb_3sec_4.fif')
data_preICA = epochs_a_preICA._data[10][12][:]

t = np.arange(2,5,3/768)
plt.figure(figsize = (8,2.5))
plt.title('C3 signal before and after ICA in HyPyP')
plt.plot(t,(pre_to_plot[0:768]-np.mean(pre_to_plot[0:768]))/np.std(pre_to_plot[0:768]),label = 'Signal after ICA')
plt.plot(t,(data_preICA-np.mean(data_preICA))/np.std(data_preICA), label = 'Signal before ICA')
plt.legend(fontsize = 9)
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('signal (' + u'\u03bc'+'V)', fontsize = 9)
plt.tight_layout()
#plt.savefig("Figures for report\\before and after ICA.png",bbox_inches='tight',dpi=300)

#%% Phase before and after ICA
freq_bands = {'Alpha' :[8, 13]}
        
sampling_rate = epochs_a_preICA.info['sfreq']
data_inter = np.array([epochs_a_preICA['Coupled'], epochs_b_preICA['Coupled']])

complex_signal, filtered = analyses.compute_freq_bands(data_inter, sampling_rate,
                                                     freq_bands)

#%% Phase after ICA
data_inter_ICA = np.array([epochs_a_4['Coupled'], epochs_b_4['Coupled']])

complex_signal_ICA, filtered_ICA = analyses.compute_freq_bands(data_inter_ICA, sampling_rate,
                                                     freq_bands)

#%% Plotting the phase comparison

#Phase
t = np.arange(2,5,3/768)
plt.figure(figsize = (8,2.5))
plt.title('C3 alpha phase before and after ICA in HyPyP')
plt.plot(t,np.angle(complex_signal[0,1,12,0,:]),label = 'Phase before ICA')
plt.plot(t,np.angle(complex_signal_ICA[0,1,12,0,:]),label = 'Phase after ICA')
#plt.plot(t,(filtered[0][0,10,12,:]-np.mean(filtered[0][0,10,12,:]))/np.std(filtered[0][0,10,12,:]),label = 'With re-referecing')
#plt.plot(t,filtered_reref[0][0,10,12,:],label = 'Without re-referencing')
plt.legend(fontsize = 9)
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('angle (radians)', fontsize = 9)
plt.tight_layout()
#plt.savefig("Figures for report\\phase_ICA.png",bbox_inches='tight',dpi=300)


#%% Comparison of first coupled epoch before and after ICA Fieldtrip
folder ='C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Connectivity-Special-Course-2\\Mat-files before ICA\\'
#first_coupled = scipy.io.loadmat(folder+'eeg_first_coupled.mat')
#first_coupled = scipy.io.loadmat(folder+'second_coupled_epoch.mat')
first_coupled = scipy.io.loadmat(folder+'pair004_subj1_preICA.mat')


keys_1 = []
for key_1 in first_coupled.keys():
    keys_1.append(key_1)
matrix_c = first_coupled[keys_1[-1]]

plt.figure(figsize = (8,2.5))
t = np.arange(2,5,3/768)
#plt.plot(t,(matrix_c[12][1280:2048]-np.mean(matrix_c[12][1280:2048]))/np.std(matrix_c[12][1280:2048]), label = 'Before ICA')
plt.plot(t,(matrix_c[0,1][0,1280:2048]-np.mean(matrix_c[0,1][0,1280:2048]))/np.std(matrix_c[0,1][0,1280:2048]), label = 'Before ICA')
#plt.plot(t,(matrix_c[0][1280:2048]-np.mean(matrix_c[0][1280:2048]))/np.std(matrix_c[0][1280:2048]), label = 'Before ICA')
plt.plot(t,(subj_1[1,12,0:768]-np.mean(subj_1[1,12,0:768]))/np.std(subj_1[1,12,0:768]), label = 'After ICA')
plt.title('Before and after ICA in Fieldtrip', fontsize = 12)
plt.legend(fontsize = 9)
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('signal (' + u'\u03bc'+'V)', fontsize = 9)
plt.tight_layout()

#%% Calculating the PLV before and after 

result,_,_,_ = analyses.compute_sync(complex_signal, mode='plv', epochs_average = False)

result_ICA,_,_,_ = analyses.compute_sync(complex_signal_ICA, mode='plv', epochs_average = False)

# Slicing of the last two dimensions
n_ch = 64

result_preICA = result[0,1,0:n_ch,n_ch:2*n_ch]
result_postICA = result_ICA[0,1,0:n_ch,n_ch:2*n_ch]

# Plotting matrices side-by-side
plt.figure(figsize=(7, 3))
plt.suptitle('Average PLV estimates from first coupled epoch of pair 4 from HyPyP', fontsize = 12)
plt.rc('font', size=8) 
plt.subplot(1,2,1)
plt.title('PLV estimates before ICA')
plt.imshow(result_preICA,cmap=plt.cm.Reds)
cbar = plt.colorbar()
cbar.set_label('PLV estimate')
plt.clim(np.min(result_postICA),np.max(result_postICA))
plt.subplot(1,2,2)
plt.title('PLV estimates after ICA')
plt.imshow(result_postICA,cmap=plt.cm.Reds)
cbar = plt.colorbar()
cbar.set_label('PLV estimate')
plt.clim(np.min(result_postICA),np.max(result_postICA))
plt.tight_layout()
plt.show()
plt.savefig("Figures for report\\HyPyP_PLV_before_and_after_ICA.png",bbox_inches='tight',dpi=300)







