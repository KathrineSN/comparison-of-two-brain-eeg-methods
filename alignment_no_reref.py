# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:37:39 2021

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

#%% Loading preprocessed (after ICA, with re-referencing)
epochs_a_reref = mne.read_epochs('Preprocessed data\\subj_a_3sec_4.fif')
epochs_b_reref = mne.read_epochs('Preprocessed data\\subj_b_3sec_4.fif')

epo_reref = epochs_a_reref['Coupled']._data

first_epoch_reref = []
for i in range(7):
    i += 1
    print(i)
    first_epoch_reref.append(epo_reref[i,12,:])
  
reref_to_plot = np.hstack(first_epoch_reref) 
#%% Loading preprocessed (after ICA, without re-referencing)
epochs_a = mne.read_epochs('Epochs without re-referencing\\subj_a_3sec_4_noref.fif')
epochs_b = mne.read_epochs('Epochs without re-referencing\\subj_b_3sec_4_noref.fif')

epo_a = epochs_a['Coupled']._data

first_epoch = []
for i in range(7):
    i += 1
    print(i)
    first_epoch.append(epo_a[i,12,:])
  
epo_a_to_plot = np.hstack(first_epoch) 

#%% Loading final received data from Marius
folder ='C:/Users/kathr/OneDrive/Documents/GitHub/Connectivity-Special-Course-2/Final mat-file/'
subj1_final = scipy.io.loadmat(folder+'eeg_pair004.mat')
# Creating list of numpy arrays from matlab file
# Subject 1
keys_1_final = []
for key in subj1_final.keys():
    keys_1_final.append(key)
subj_1_final = subj1_final[keys_1_final[-1]]

#%% Comrison between HyPyP and Fieldtrip after re-referencing
t = np.arange(2,5,3/768)
plt.figure(figsize = (8,2.5))
plt.title('C3 signal after re-referencing')
plt.plot(t,(reref_to_plot[0:768]-np.mean(reref_to_plot[0:768]))/np.std(reref_to_plot[0:768]),label = 'HyPyP')
plt.plot(t,(subj_1_final[2,0:768]-np.mean(subj_1_final[2,0:768]))/np.std(subj_1_final[2,0:768]), label = 'Fieldtrip')
plt.legend(fontsize = 9)
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('signal (' + u'\u03bc'+'V)', fontsize = 9)
plt.tight_layout()
plt.savefig("Figures for report\\signal_after_reref.png",bbox_inches='tight',dpi=300)

#%% Plot with and without re-referencing

t = np.arange(2,5,3/768)
plt.figure(figsize = (8,2.5))
plt.title('C3 signal before and after re-referencing in HyPyP')
plt.plot(t,(reref_to_plot[0:768]-np.mean(reref_to_plot[0:768]))/np.std(reref_to_plot[0:768]),label = 'signal after re-referecing')
plt.plot(t,(epo_a_to_plot[0:768]-np.mean(epo_a_to_plot[0:768]))/np.std(epo_a_to_plot[0:768]),label = 'signal before re-referencing')
plt.legend(fontsize = 9)
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('signal (' + u'\u03bc'+'V)', fontsize = 9)
plt.tight_layout()
plt.savefig("Figures for report\\signal_reref.png",bbox_inches='tight',dpi=300)

#%% Comparison of phase estimate 
freq_bands = {'Alpha' :[8, 13]}
        
sampling_rate = epochs_a.info['sfreq']
data_inter = np.array([epochs_a['Coupled'], epochs_b['Coupled']])

complex_signal, filtered = analyses.compute_freq_bands(data_inter, sampling_rate,
                                                     freq_bands)

#%%
data_inter_reref = np.array([epochs_a_reref['Coupled'], epochs_b_reref['Coupled']])

complex_signal_reref, filtered_reref = analyses.compute_freq_bands(data_inter_reref, sampling_rate,
                                                     freq_bands)

#%% Plot the phases Epoch no. 10

#alpha
t = np.arange(2,5,3/768)
plt.figure(figsize = (8,2.5))
plt.title('Alpha band of C3 Before and after ICA in HyPyP')
plt.plot(t,filtered[0][0,1,12,:],label = 'With re-referecing')
plt.plot(t,filtered_reref[0][0,1,12,:],label = 'Without re-referencing')
plt.legend(fontsize = 9)
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('signal (' + u'\u03bc'+'V)', fontsize = 9)
plt.tight_layout()
#plt.savefig("Figures for report\\alpha_reref.png",bbox_inches='tight',dpi=1200)

#%%
#Phase
t = np.arange(2,5,3/768)
plt.figure(figsize = (8,2.5))
plt.title('C3 alpha phase before and after re-referencing in HyPyP')
plt.plot(t,np.angle(complex_signal[0,1,12,0,:]),label = 'signal before re-referecing')
plt.plot(t,np.angle(complex_signal_reref[0,1,12,0,:]),label = 'signal after re-referecing')
#plt.plot(t,(filtered[0][0,10,12,:]-np.mean(filtered[0][0,10,12,:]))/np.std(filtered[0][0,10,12,:]),label = 'With re-referecing')
#plt.plot(t,filtered_reref[0][0,10,12,:],label = 'Without re-referencing')
plt.legend(fontsize = 9)
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('angle (radians)', fontsize = 9)
plt.tight_layout()
#plt.savefig("Figures for report\\phase_reref.png",bbox_inches='tight',dpi=300)

#%% Compare phase between HyPyP and Fieldtrip before and after ICA

#Phase
t = np.arange(2,5,3/768)
plt.figure(figsize = (8,2.5))
plt.title('C3 alpha phase after re-referencing')
plt.plot(t,np.angle(complex_signal_reref[0,1,12,0,:]),label = 'HyPyP')
plt.plot(t,subj_1_final[4,0:768],label = 'Fieldtrip')
plt.legend(fontsize = 9)
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('angle (radians)', fontsize = 9)
plt.tight_layout()
plt.savefig("Figures for report\\phase_after_reref.png",bbox_inches='tight',dpi=300)


#%%Plotting the PLV values before and after re-referencing

result,_,_,_ = analyses.compute_sync(complex_signal, mode='plv', epochs_average = False)

result_reref,_,_,_ = analyses.compute_sync(complex_signal_reref, mode='plv', epochs_average = False)

# Slicing of the last two dimensions
n_ch = 64

result_pre_reref = result[0,1,0:n_ch,n_ch:2*n_ch]
result_post_reref = result_reref[0,1,0:n_ch,n_ch:2*n_ch]

# Plotting matrices side-by-side
plt.figure(figsize=(7, 3))
plt.suptitle('Average PLV estimates from first coupled epoch of pair 4 from HyPyP', fontsize = 12)
plt.rc('font', size=8) 
plt.subplot(1,2,1)
plt.title('PLV estimates before re-referencing')
plt.imshow(result_pre_reref,cmap=plt.cm.Reds)
cbar = plt.colorbar()
cbar.set_label('PLV estimate')
plt.clim(np.min(result_pre_reref),np.max(result_pre_reref))
plt.subplot(1,2,2)
plt.title('PLV estimates after re-referencing')
plt.imshow(result_post_reref,cmap=plt.cm.Reds)
cbar = plt.colorbar()
cbar.set_label('PLV estimate')
plt.clim(np.min(result_pre_reref),np.max(result_pre_reref))
plt.tight_layout()
plt.show()
plt.savefig("Figures for report\\HyPyP_PLV_before_and_after_reref.png",bbox_inches='tight',dpi=300)


#%%Plotting PLV after re-referencing for the first coupled epoch
folder ='C:/Users/kathr/OneDrive/Documents/GitHub/Connectivity-Special-Course-2/Final phase estimates/'
subj1_final = scipy.io.loadmat(folder+'phaseestimates.mat')

# Creating list of numpy arrays from matlab file
# Subject 1
keys_1_final = []
for key in subj1_final.keys():
    keys_1_final.append(key)
phases = subj1_final[keys_1_final[-1]]

s1_ft = phases[0,0:768]
s2_ft = phases[1,0:768]

s1_hypyp = np.angle(complex_signal_reref[0,1,12,0,:])
s2_hypyp = np.angle(complex_signal_reref[1,1,12,0,:])



def plv(x1,x2):
    phase_diff = x1-x2
    con = abs(np.mean(np.exp(1j*phase_diff)))
    
    return con

con_pypyp = plv(s1_hypyp,s2_hypyp)
con_ft = plv(s1_ft,s2_ft)










