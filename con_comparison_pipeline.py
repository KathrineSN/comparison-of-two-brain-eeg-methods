# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 09:15:26 2021

@author: kathr
"""
import os
import mne
import numpy as np
from plot_functions import heatmap
import matplotlib.pyplot as plt
import scipy.io
import re

# Setting path
path="C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Connectivity-Special-Course-2"
os.chdir(path)

#%% Opening matlab matrices
folder ='C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Connectivity-Special-Course-2\\Connectivity matrices\\plv\\plv per pair from Marius 29-11-21\\'

matrices = []
files = os.listdir(folder)

for f in files:
    if f.startswith('IBSavg'):
        print(f)
        mat_file = scipy.io.loadmat(folder + f)
        f_name = f[0:32]
        keys = []
        for key in mat_file.keys():
            keys.append(key)
        con_matrix = mat_file[keys[-1]]
        
        # Storing matrices in a list
        matrices.append(con_matrix)
        
        # Plotting each matrix
        fig, ax = plt.subplots(figsize=(16,16))
        plt.suptitle(f_name,fontsize = 25)
        montage = mne.channels.make_standard_montage("biosemi64")
        new_ch_names = montage.ch_names
        im, cbar = heatmap(con_matrix, new_ch_names, new_ch_names, ax=ax,
                       cmap=plt.cm.Reds, cbarlabel='plv')
        plt.show()
        
        # Saving images of matrices
        savepath = 'C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Connectivity-Special-Course-2\\Connectivity matrices\\plv\\images\\average pair matrices from Marius - 29-11-21\\'
        #plt.savefig(savepath + f_name + '.png')
        
plt.close('all')
#%% Loading 3sec matrices
folder_2 ='C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Connectivity-Special-Course-2\\Connectivity matrices\\plv\\plv per pair\\'

matrices_2 = []

files_2 = os.listdir(folder_2)

for f in files_2:
    if f.endswith('alpha_Coupled_3sec.npy'):
        print(f)
        matrix = np.load(folder_2 + f)   
        matrices_2.append(matrix)
        f_name = f[0:30]
        fig, ax = plt.subplots(figsize=(16,16))
        plt.suptitle(f_name,fontsize = 25)
        montage = mne.channels.make_standard_montage("biosemi64")
        new_ch_names = montage.ch_names
        im, cbar = heatmap(matrix, new_ch_names, new_ch_names, ax=ax,
                       cmap=plt.cm.Reds, cbarlabel='plv')
        plt.show()
        
        # Saving images of matrices
        savepath = 'C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Connectivity-Special-Course-2\\Connectivity matrices\\plv\\images\\average pair matrices - 29-11-21\\'
        #plt.savefig(savepath + f_name + '.png')

print(len(matrices_2))

plt.close('all')
#%% Loading segmented 3sec matrices
folder_3 ='C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Connectivity-Special-Course-2\\Connectivity matrices\\plv\\plv per pair segmented 25s epochs - 02-12-21\\'

matrices_3 = []

files_3 = os.listdir(folder_3)

for f in files_3:
    if f.endswith('alpha_Coupled_3sec(25).npy'):
        print(f)
        matrix = np.load(folder_3 + f)   
        matrices_3.append(matrix)

print(len(matrices_3))


#%% Rename files 
for f in files_2:
    if f.find('001') != -1 or f.find('002') != -1:
        new_name = f.split('00')[0]+'0'+f.split('00')[1]
        os.rename(os.path.join(folder_2,f), os.path.join(folder_2,new_name))

#%% Calculate correlation between the two sets of matrices
from scipy.stats import pearsonr
corrs = []
corrs2 = []
ids = [3,4,5,7,9,10,11,12,14,16,17,18,19,20,22,23,24,27]
for i in range(18):
    m1 = matrices[i].flatten()
    m2 = matrices_2[i].flatten()
    

    corrs2.append(pearsonr(m1,m2))
    corrs.append(np.corrcoef(m1,m2)[0,1])
    print('Pair number %d \n' %ids[i])
    print('Matlab: \n min: %f \n max: %f \n HyPyP: \n min: %f \n max: %f \n' %(np.min(m1),np.max(m1),np.min(m2),np.max(m2)))

print('correlations between matrices:')
print(corrs)

#%% Plotting average matrices next to each other
avg_ft = np.sum(matrices, axis = 0)/len(matrices)
avg_hypyp = np.sum(matrices_2, axis = 0)/len(matrices_2)
ft_min = np.min(np.array(matrices))
ft_max = np.max(np.array(matrices))
hypyp_min = np.min(np.array(matrices_2))
hypyp_max = np.max(np.array(matrices_2))
print('Overall min and max values')
print('Matlab: \n min: %f \n max: %f \n HyPyP: \n min: %f \n max: %f \n' %(ft_min,ft_max,hypyp_min,hypyp_max))


plt.figure(figsize=(7, 3))
plt.suptitle('Average PLV estimates from HyPyP and Fieldtrip', fontsize = 12)
plt.rc('font', size=8) 
plt.subplot(1,2,1)
plt.title('Fieldtrip based PLV estimate')
plt.imshow(avg_ft,cmap=plt.cm.Reds)
cbar = plt.colorbar()
cbar.set_label('PLV estimate')
plt.clim(np.min(avg_ft),np.max(avg_hypyp))
plt.subplot(1,2,2)
plt.title('HyPyP based PLV estimate')
plt.imshow(avg_hypyp,cmap=plt.cm.Reds)
cbar = plt.colorbar()
cbar.set_label('PLV estimate')
plt.clim(np.min(avg_ft),np.max(avg_hypyp))
plt.tight_layout()
plt.show()
#plt.savefig("Figures for report\\HyPyP_Ft_Comparison.png",bbox_inches='tight',dpi=1200)
  
#%% Plotting average matrices for all three approaches

import matplotlib.ticker as tick
avg_fieldtrip_mat = np.sum(matrices, axis = 0)/len(matrices)
avg_hypyp3_mat = np.sum(matrices_2, axis = 0)/len(matrices_2)
avg_hypypseg_mat = np.sum(matrices_3, axis = 0)/len(matrices_3)

plt.figure(figsize=(10, 3))
plt.suptitle('Phase Locking Value', fontsize = 12)
plt.rc('font', size=8) 
plt.subplot(1,3,1)
plt.title('Fieldtrip estimate')
plt.imshow(avg_fieldtrip_mat,cmap=plt.cm.Reds)
cbar = plt.colorbar()
cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
plt.clim(np.min(avg_fieldtrip_mat),np.max(avg_hypyp3_mat))
plt.subplot(1,3,2)
plt.title('HyPyP epoching before filter')
plt.imshow(avg_hypyp3_mat,cmap=plt.cm.Reds)
cbar = plt.colorbar()
cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
plt.clim(np.min(avg_fieldtrip_mat),np.max(avg_hypyp3_mat))
plt.subplot(1,3,3)
plt.title('HyPyP epoching after filter')
plt.imshow(avg_hypypseg_mat,cmap=plt.cm.Reds)
cbar = plt.colorbar()
cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
plt.clim(np.min(avg_fieldtrip_mat),np.max(avg_hypyp3_mat))
plt.show()
#plt.savefig("Figures for report\\epoch_after_filter.png",bbox_inches='tight',dpi=300)

#%% Correlations between the 3 approaches
corrs_hypyp = []
corrs_hypyp_ft = []
ids = [3,4,5,7,9,10,11,12,14,16,17,18,19,20,22,23,24,27]
for i in range(18):
    m1 = matrices[i].flatten()
    m2 = matrices_2[i].flatten()
    m3 = matrices_3[i].flatten()

    corrs_hypyp.append(np.corrcoef(m2,m3)[0,1])
    corrs_hypyp_ft.append(np.corrcoef(m1,m3)[0,1])
    #print('Pair number %d \n' %ids[i])
    #print('HyPyP: \n min: %f \n max: %f \n HyPyP seg: \n min: %f \n max: %f \n' %(np.min(m1),np.max(m1),np.min(m2),np.max(m2)))

print('correlations between hypyp matrices:')
print(corrs_hypyp)
print('min and max for correlations between hypyp matrices:')
print(np.min(np.array(corrs_hypyp)))
print(np.max(np.array(corrs_hypyp)))

print('correlations between hypyp seg and ft matrices:')
print(corrs_hypyp_ft)
print('min and max for correlations between hypyp and fieldtrip matrices:')
print(np.min(np.array(corrs_hypyp_ft)))
print(np.max(np.array(corrs_hypyp_ft)))


print('min and max for average matrices')
print('fieldtrip')
print(np.min(np.array(matrices)))
print(np.max(np.array(matrices)))
print('hypyp')
print(np.min(np.array(matrices_2)))
print(np.max(np.array(matrices_2)))
print('hypyp seg')
print(np.min(np.array(matrices_3)))
print(np.max(np.array(matrices_3)))




















        