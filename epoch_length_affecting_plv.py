# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:38:00 2021

@author: kathr
"""

# Importing dependencies
import os
path="C:\\Users\\kathr\\OneDrive\\Documents\\Github\\Connectivity-Special-Course-2"
os.chdir(path)
import mne
import numpy as np
import matplotlib.pyplot as plt

#%% Loading matrices related to each epoch length

matrices_1_sec = []
matrices_3_sec = []
matrices_25_sec = []

folder = 'C:\\Users\\kathr\\OneDrive\Documents\\GitHub\\Connectivity-Special-Course-2\\Connectivity matrices\\plv\\plv per pair\\'

files = os.listdir(folder)

for f in files:
    if f.endswith('alpha_Coupled_1sec.npy'):
        matrices_1_sec.append(np.load(folder + f))
        
for f in files:
    if f.endswith('alpha_Coupled_3sec.npy'):
        matrices_3_sec.append(np.load(folder + f))
        
for f in files:
    if f.endswith('alpha_Coupled_25sec.npy'):
        matrices_25_sec.append(np.load(folder + f))

avg_1_sec = np.mean(np.array(matrices_1_sec))
avg_3_sec = np.mean(np.array(matrices_3_sec))
avg_25_sec = np.mean(np.array(matrices_25_sec))
plvs = np.array([avg_1_sec, avg_3_sec, avg_25_sec])
errors = np.array([np.std(matrices_1_sec)/np.sqrt(18), np.std(matrices_3_sec)/np.sqrt(18),np.std(matrices_25_sec)/np.sqrt(18)])

#%% plot

epo_length = np.array([1,3,25])
fig, ax = plt.subplots(figsize=(7,3))
#plt.figure(figsize=(8, 3))
ax.plot(epo_length,plvs, label = 'Estimate')
ax.fill_between(epo_length, (plvs-errors), (plvs+errors), color='b', alpha=.1, label = 'Error')
plt.title('Average PLV Variability over Epoch Length')
plt.xlabel('Epoch Length')
plt.ylabel('Average PLV')
plt.legend()
plt.tight_layout()
#plt.savefig(plt.savefig("Figures for report\\Epoch_length_affecting_plv.png", dpi=1200))

#%% Alternative plot
plt.figure(figsize=(7, 3))
plt.title('Average PLV variability over Epoch Length')
#plt.subplot(1,2,1)
plt.xlabel('Epoch Length')
plt.ylabel('Average PLV')
plt.errorbar(epo_length, plvs, yerr = errors, ecolor = 'red', elinewidth=(2.5), label = 'error')
plt.tight_layout()

#%% Plot the average matrix for the three epoch lengths
import matplotlib.ticker as tick
avg_1_mat = np.sum(matrices_1_sec, axis = 0)/len(matrices_1_sec)
avg_3_mat = np.sum(matrices_3_sec, axis = 0)/len(matrices_3_sec)
avg_25_mat = np.sum(matrices_25_sec, axis = 0)/len(matrices_25_sec)

plt.figure(figsize=(10, 3))
plt.suptitle('Phase Locking Value', fontsize = 12)
plt.rc('font', size=8) 
plt.subplot(1,3,1)
plt.title('Average 1 seconds estimate')
plt.imshow(avg_1_mat,cmap=plt.cm.Reds)
cbar = plt.colorbar()
cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
plt.subplot(1,3,2)
plt.title('Average 3 seconds estimate')
plt.imshow(avg_3_mat,cmap=plt.cm.Reds)
cbar = plt.colorbar()
cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
plt.subplot(1,3,3)
plt.title('Average 21 seconds estimate')
plt.imshow(avg_25_mat,cmap=plt.cm.Reds)
cbar = plt.colorbar()
cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))
plt.show()
plt.savefig("Figures for report\\length_matrix_plv.png",bbox_inches='tight',dpi=300)

#%% Create boxplot for the three lengths 

data = [np.array(matrices_1_sec).flatten(), np.array(matrices_3_sec).flatten(), np.array(matrices_25_sec).flatten()]
fig, ax = plt.subplots()
fig.set_size_inches(6, 4)
ax.set_title('Distribution of PLV estimates for the 3 epoch lengths', fontsize = 12)
ax.boxplot(data, showfliers=False, labels = ('1','3','21'))
ax.set_xlabel('Epoch length',fontsize = 9)
ax.set_ylabel('PLV estimate',fontsize = 9)
plt.tight_layout()
plt.show()
plt.savefig("Figures for report\\epoch_length_boxplot.png",bbox_inches='tight',dpi=300)










