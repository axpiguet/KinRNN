#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import numpy as np
import pandas as pd
import _pickle as cPickle
import torch
import matplotlib.pyplot as plt
import data
from data import stim
import rnn
import utils as utils
import tests.params_files
from importlib import import_module, __import__
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from utils import plot_electrode_activation
import _pickle as cPickle
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import List, Tuple
from data import PATH, N_ELECTRODES, MUSCLES
import scipy.io
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

emg = cPickle.load(open("emg4.pkl", "rb" ))
kin = cPickle.load(open("kin4.pkl", "rb" ))

# fs vicon : 100 Hz       fs emg = 1259.25925925926 Hz

emg_data =  np.expand_dims(np.array(list(emg[emg.columns[-13:]].iloc[0,:].values))[:,:1134], axis = 0)
for i in range (1,emg.shape[0]):

    emg_data = np.concatenate((emg_data, np.expand_dims(np.array(list(emg[emg.columns[-13:]].iloc[i,:].values))[:,:1134], axis = 0)), axis=0)
print(emg_data.shape)

# NORMALIZATION
emg_plot = emg_data
for musc in range (emg_data.shape[1]):
    min_ = np.nanmin(emg_data[:,musc,:])
    max_ = np.nanmax(emg_data[:,musc,:])
    emg_plot[:,musc,:] = (2*(emg_data[:,musc,:] - min_) / (max_ - min_))-1
emg_plot = 100*emg_plot

#####################################
for i in range(emg.shape[0]):
    selected_trial = i#162
    emg_trial = emg_plot[selected_trial, :,:]
    fsemg = 1259.25925925926
    fskin = 100
    duration_ms = int(np.ceil(1000*emg_trial.shape[1] / fsemg))
    from sklearn.utils import resample
    emg_trial = resample(emg_trial.T, n_samples=kin.Frame.iloc[selected_trial].shape[0], random_state=0)

    ##########

    fig, ax = plt.subplots(3,2 , figsize= (13,10),  gridspec_kw={'height_ratios':[1, 5,3]})

    stim = data.create(emg[['Frequency','Pulses','PulseWidth', 'Amplitude','Anodes', 'Cathodes']] , duration_ms, 10*fskin)
    time_forstim = np.linspace(0, stim.shape[2]-1,stim.shape[2])
    #stim = data.create(emg[['Frequency','Pulses','PulseWidth', 'Amplitude','Anodes', 'Cathodes']] , duration_ms, fskin)#, duration in ms, fs in hz)
    time = np.linspace(0, emg_trial.shape[0]-1,emg_trial.shape[0])
    print(stim.shape)
    ax[0, 0].set_title('LEFT', fontweight='bold')
    ax[0, 1].set_title('RIGHT', fontweight='bold')
    ax[0][0].set_yticks([])
    ax[0][0].set_ylim([-4,3])
    ax[0][1].set_yticks([])
    ax[0][1].set_xticks([])
    ax[0][0].set_xticks([])
    ax[0][1].set_ylim([-5,3])
    ax[0][0].set_frame_on(False)
    ax[0][1].set_frame_on(False)
    #plot the stim
    ax[0][0].plot(time_forstim, stim[selected_trial,0,:,:], color='#fa525b')
    ax[0][1].plot(time_forstim, stim[selected_trial,0,:,:], color='#fa525b')
    ax[0][1].set_title(str(emg.iloc[selected_trial:selected_trial+1].Pulses.values[0])+' pulses      ', fontsize=12, loc='right')
    #ax[0][0].plot(time[0:stim.shape[2]], stim[selected_trial,0,:,:], color='#fa525b')
    #ax[0][1].plot(time[0:stim.shape[2]], stim[selected_trial,0,:,:], color='#fa525b')
    #plot the EMG
    for side in [0,1]:

        for r in range(7):
            if not (side and r ==5):
                #predicted_line = ax[1][side].plot(time, emg_trial[:,r-1+7*(side)]-r*150, linewidth=1, color = 'lightgrey')
                predicted_line = ax[1][side].plot(time, emg_trial[:,r-1+7*(side)]-r*230, linewidth=1, color = 'lightgrey')

            ax[1][side].set_frame_on(False)
            #ax[1][side].set_ylim([-1000,50])
            ax[1][side].set_ylim([-1500,130])
            ax[1][side].set_xticks([])
            labs = ['100','0','-100','100','0','-100','100','0','-100','100','0','-100','100','0','-100','100','0','-100','100','0','-100',]
            #ax[1][side].tick_params('x', labelbottom=False)
            ax[1][0].yaxis.set_minor_locator(ticker.FixedLocator([-1480,-1380,-1280,-1250,-1150,-1050,-1020,-920,-820,-790,-690,-590,-560,-460,-360,-330,-230,-130,-100,0,100]))
            ax[1][0].yaxis.set_minor_formatter(ticker.FixedFormatter(labs))
            ax[1][0].tick_params(which='minor', length=3, labelcolor = 'silver', labelsize = 'x-small')

            ax[1][side].tick_params('x', labelbottom=True, labelsize="10")
        #ax[1][side].set_xlabel('Time [ms]', fontsize="10")
    #ax[1][0].set_yticks([-900 , -750, -600,-450,-300, -150, 0], labels=('Sol','TA','MG','ST','VLat','RF','Add'))
    #ax[1][0].set_yticks([-1380 , -1150, -920,-690,-460, -230, 0], labels=('Sol','TA','MG','ST','VLat','RF','Add'))
    #ax[1][0].tick_params(which='major', length=18, labelsize = 'large', color = 'k')
    loc = [-1380 , -1150, -920,-690,-460, -230, 0]
    labels=['Sol','TA','MG','ST','VLat','RF','Add']
    for i in range(7):
        ax[1][0].text(-13, loc[i]-20, labels[i], fontsize = 'large')
    ax[1][0].set_title('Muscles', fontsize=12, loc='left')
    ax[1][1].set_yticks([])
    ax[1][0].set_yticks([])
    ax[0][1].set_yticks([])

    #plot the EMG
    letter = ['L','R']
    for side in [0,1]:
        # plot z coordinate
        ax[2][0].set_title('Ankle and Knee height', fontsize=12, loc='left')
        ax[2][side].set_frame_on(False)
        ax[2][side].plot(time, kin[letter[side]+'ANK'].iloc[selected_trial][:,2], color = 'darkorange' , linewidth=1)
        ax[2][side].plot(time, kin[letter[side]+'KNE'].iloc[selected_trial][:,2] , color = 'orchid', linewidth=1)
        ax[2][side].set_xlabel('Time [ms]', fontsize="12")
        ax[2][side].tick_params(colors = 'silver')
        ax[2][1].set_ylim([300,1400])
        ax[2][0].set_ylim([300,1400])
        ax[2][1].set_yticks([])
    plt.tight_layout()
    plt.show()
    f = "C:/Users/yes/Documents/GitHub/little_RNN/src/observedata/trial" +str(selected_trial)+".png"
    fig.savefig(f)
    print(f)
