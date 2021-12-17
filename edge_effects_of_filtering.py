# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:55:29 2021

@author: kathr
"""

# Import dependecies
import os
path="C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Connectivity-Special-Course-2\\Effect of filtering\\"
os.chdir(path)
import mne
import numpy as np
import matplotlib.pyplot as plt

#%% Loading data before and after filtering
epo_a_c = mne.read_epochs('epo_bf_coupled_4.fif')
freq_data = np.load('filtered_coupled_4.npy')

#%% Extracting data from the first 5 epochs and channel C3
epo_coupled_data = epo_a_c._data
coupled_to_plot = np.hstack((epo_coupled_data[0,12,:],epo_coupled_data[1,12,:],epo_coupled_data[2,12,:],epo_coupled_data[3,12,:],epo_coupled_data[4,12,:]))

filtered = freq_data[1][0,0:9,12,:]
filtered_to_plot = np.hstack((filtered[0],filtered[1],filtered[2],filtered[3],filtered[4]))

#%% Plotting the edges between epochs

plt.figure(figsize = (8,5))
plt.subplot(2,2,1)
plt.title('Edge between epoch 1 and 2')
x = np.linspace(718/256,818/256,100)
plt.plot(x,coupled_to_plot[718:818], label = 'C3 signal before filtering')
plt.plot(x,filtered_to_plot[718:818], label = 'C3 alpha band after filtering')
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('signal (' + u'\u03bc'+'V)', fontsize = 9)
plt.legend(fontsize = 9)

plt.subplot(2,2,2)
plt.title('Edge between epoch 2 and 3')
x = np.linspace(1486/256,1586/256,100)
plt.plot(x,coupled_to_plot[1486:1586], label = 'before filtering')
plt.plot(x,filtered_to_plot[1486:1586], label = 'alpha band after filtering')
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('signal (' + u'\u03bc'+'V)', fontsize = 9)

plt.subplot(2,2,3)
plt.title('Edge between epoch 3 and 4')
x = np.linspace(2254/256,2354/256,100)
plt.plot(x,coupled_to_plot[2254:2354], label = 'before filtering')
plt.plot(x,filtered_to_plot[2254:2354], label = 'alpha band after filtering')
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('signal (' + u'\u03bc'+'V)', fontsize = 9)

plt.subplot(2,2,4)
plt.title('Edge between epoch 4 and 5')
x = np.linspace(3022/256,3122/256,100)
plt.plot(x,coupled_to_plot[3022:3122], label = 'C3 signal before filtering')
plt.plot(x,filtered_to_plot[3022:3122], label = 'C3 alpha band after filtering')
plt.tight_layout()
plt.xlabel('time (s)', fontsize = 9)
plt.ylabel('signal (' + u'\u03bc'+'V)', fontsize = 9)
plt.savefig("C:\\Users\\kathr\\OneDrive\\Documents\\GitHub\\Connectivity-Special-Course-2\\Figures for report\\filtering_effects.png",bbox_inches='tight',dpi=300)


