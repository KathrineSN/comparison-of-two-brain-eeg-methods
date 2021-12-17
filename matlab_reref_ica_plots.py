# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 18:39:55 2021

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

#%% Loading the data
folder ='C:/Users/kathr/OneDrive/Documents/GitHub/Connectivity-Special-Course-2/Final mat-file/'
subj1_final = scipy.io.loadmat(folder+'eeg_pair004.mat')

# Creating list of numpy arrays from matlab file
# Subject 1
keys_1_final = []
for key in subj1_final.keys():
    keys_1_final.append(key)
subj_1_final = subj1_final[keys_1_final[-1]]

#%% Plotting

# Before and after ICA
t = np.arange(2,5,3/768)
plt.figure(figsize = (8,2.5))
plt.title('C3 signal before and after ICA in Fieldtrip', fontsize = 12)
plt.plot(t,(subj_1_final[0,0:768]-np.mean(subj_1_final[0,0:768]))/np.std(subj_1_final[0,0:768]),label = 'Signal before ICA')
plt.plot(t,(subj_1_final[1,0:768]-np.mean(subj_1_final[1,0:768]))/np.std(subj_1_final[1,0:768]),label = 'Signal after ICA')
plt.legend(fontsize = 9)
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('angle (radians)', fontsize = 9)
plt.tight_layout()
plt.savefig("Figures for report\\Ft_before_after_ICA.png",bbox_inches='tight',dpi=300)

# Before and after re-referencing
t = np.arange(2,5,3/768)
plt.figure(figsize = (8,2.5))
plt.title('C3 signal before and after re-referencing in Fieldtrip', fontsize = 12)
plt.plot(t,(subj_1_final[1,0:768]-np.mean(subj_1_final[1,0:768]))/np.std(subj_1_final[1,0:768]),label = 'Signal before re-referencing')
plt.plot(t,(subj_1_final[2,0:768]-np.mean(subj_1_final[2,0:768]))/np.std(subj_1_final[2,0:768]),label = 'Signal after re-referencing')
plt.legend(fontsize = 9)
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('angle (radians)', fontsize = 9)
plt.tight_layout()
plt.savefig("Figures for report\\Ft_before_after_reref.png",bbox_inches='tight',dpi=300)





