import argparse
import numpy as np
import pandas as pd
import _pickle as cPickle
import torch
import matplotlib.pyplot as plt
import data
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
from matplotlib.patches import Rectangle
import scipy.io
import math
from sklearn.utils import resample
from scipy import interpolate
from scipy.signal import find_peaks
###################################################


#selected_trial = 54
###################################################


emgold = cPickle.load(open("processednot_norm_emg1.pkl", "rb" ))
kin = cPickle.load(open("processed_kin1.pkl", "rb" ))
stim = cPickle.load(open("processed_stim1.pkl", "rb" ))

fsemg = 1259.25925925926
fskin = 148.148
stimfs = 10*fskin

def normalize(emg_df, cols_to_cropreplace, emgfs):
    nbpt_remove = int(emgfs*0.4)
    for muscle in cols_to_cropreplace :
        rows = []
        max_ = []
        for row in emg_df[muscle].values:
            rows.append(100*row[:])# / np.max(np.abs(row[nbpt_remove:])))
            max_.append(np.max(np.abs(row[nbpt_remove:])))
        del emg_df[muscle]
        emg_df[muscle] = rows / np.max(max_)
    return emg_df


def get_data(selected_trial, LEGS2 = False):
    emg =  normalize(emgold, list(emgold.columns[-14:]), fsemg)
    #selected_trial = emg.loc[((emg['Cathodes'].astype(str) == '[3]') & (emg['Frequency'] == 20)& (emg['Amplitude'] == 3.0))].index.values[0]
    #selected_trial = emg.loc[((emg['Cathodes'].astype(str) == '[11]') &(emg['Anodes'].astype(str) == '[13]') & (emg['Frequency'] == 40)& (emg['Amplitude'] == 4.0))].index.values[0]
     ### final
    #selected_trial = emg.loc[((emg['Cathodes'].astype(str) == '[5]') &(emg['Anodes'].astype(str) == '[4]') & (emg['Frequency'] == 20)& (emg['Amplitude'] == 1.5))].index.values[0]

    #selected_trial = emg.loc[((emg['Cathodes'].astype(str) == '[5]') &(emg['Anodes'].astype(str) == '[7]') & (emg['Frequency'] == 120)& (emg['Amplitude'] == 1.0))].index.values[0]
    #selected_trial = emg.loc[((emg['Cathodes'].astype(str) == '[4]')  & (emg['Frequency'] == 20)& (emg['Amplitude'] == 3.5))].index.values[0]
    #selected_trial = emg.loc[((emg['Cathodes'].astype(str) == '[5]') &(emg['Anodes'].astype(str) == '[4, 6, 14, 15]')  & (emg['Frequency'] == 120)& (emg['Amplitude'] == 3.0))].index.values[0]
    #selected_trial = emg.loc[((emg['Cathodes'].astype(str) == '[4]') &(emg['Anodes'].astype(str) == '[3, 5]')  & (emg['Frequency'] == 20)& (emg['Amplitude'] == 3.5))].index.values[0]
    #selected_trial = emg.loc[((emg['Cathodes'].astype(str) == '[4]') &(emg['Anodes'].astype(str) == '[3, 5]')  & (emg['Frequency'] == 80)& (emg['Amplitude'] == 2.0))].index.values[0]
    #selected_trial = emg.loc[((emg['Cathodes'].astype(str) == '[10]') &(emg['Anodes'].astype(str) == '[16]')  & (emg['Frequency'] == 120)& (emg['Amplitude'] == 3.5))].index.values[0]

    #print(emg.loc[((emg['Cathodes'].astype(str) == '[10]') &(emg['Anodes'].astype(str) == '[16]')  & (emg['Frequency'] == 120)& (emg['Amplitude'] == 3.5))])
    print(emg.iloc[selected_trial])
    #print(emg.loc[((emg['Cathodes'].astype(str) == '[11]') &(emg['Anodes'].astype(str) == '[13]') & (emg['Frequency'] == 40)& (emg['Amplitude'] == 4.0))])
    #print(emg.iloc[selected_trial])
    #print(emg.iloc[((emg['Cathodes'].astype(str) == '[3]') & (emg['Frequency'] == 20)& (emg['Amplitude'] == 3.0))].index.values[0])
    #emg.iloc[selected_trial].to_pickle("goodEMGoverfit.pkl")
    #kin.iloc[selected_trial].to_pickle("goodKINoverfit.pkl")
    #stim.iloc[selected_trial].to_pickle("goodSTIMoverfit.pkl")


    #print(emg.loc[((emg['Cathodes'].astype(str) == '[3]') & (emg['Frequency'] == 20)& (emg['Amplitude'] == 3.0))])
    #print(emg.iloc[ind])
    ###################################################

    def to_coo (HIPF, ADD, KNEX):
        KNEX = KNEX + -70
        HIPF = HIPF +5
        l_thigh = 23; l_shank = 25;
        coo_hip = []
        coo_knee = []
        coo_ankle = []
        for t in range(len(HIPF)):
              knee_x = l_thigh*np.cos(np.radians(HIPF[t]))*np.cos(np.radians(ADD[t]));
              knee_y = l_thigh*np.cos(np.radians(HIPF[t]))*np.sin(np.radians(ADD[t]));
              knee_z = l_thigh*np.sin(np.radians(HIPF[t]));
              l_hiptoankle = np.sqrt(l_thigh**2 + l_shank**2 - 2*l_thigh*l_shank*np.cos(np.radians(180-KNEX[t])));
              theta = math.degrees(math.acos((l_thigh**2 + l_hiptoankle**2 - l_shank**2)/(2*l_thigh*l_hiptoankle)));
              alpha = HIPF[t] - theta;
              ankle_x = l_hiptoankle*np.cos(np.radians(alpha))*np.cos(np.radians(ADD[t]));
              ankle_y = l_hiptoankle*np.cos(np.radians(alpha))*np.sin(np.radians(ADD[t]));
              ankle_z = l_hiptoankle*np.sin(np.radians(alpha));
              coo_hip.append(np.array([0, 0, 0]));
              coo_knee.append(np.array([knee_x, knee_y, knee_z]));
              coo_ankle.append(np.array([ankle_x, ankle_y, ankle_z]));
        return np.array(coo_hip), np.array(coo_knee) , np.array(coo_ankle)

    #lhip, lknee, lank = to_coo(kin.LHIPF.iloc[selected_trial], kin.LAdd.iloc[selected_trial],kin.LKNEX.iloc[selected_trial] )

    # functions for plotting
    from utils import plot_electrode_activation
    import matplotlib


    ###################################################
    # SETTINGS

    # for electrode
    from utils import ELECTRODE_POSITIONS_ELEC
    with open(os.path.abspath( '../images/electrode.png'), 'rb') as electrode_file:
        electrode_im = plt.imread(electrode_file)
    height, width = electrode_im.shape[0], electrode_im.shape[1]
    x_offset = 4
    y_offset = 90 #165
    x_anodes, y_anodes = [], []
    for anode in kin['Anodes'].iloc[selected_trial]:
        x_anodes.append(ELECTRODE_POSITIONS_ELEC[anode][0]*width+x_offset)
        y_anodes.append(ELECTRODE_POSITIONS_ELEC[anode][1]*height+y_offset)
    x_cathodes, y_cathodes = [], []
    for cathode in kin['Cathodes'].iloc[selected_trial]:
        x_cathodes.append(ELECTRODE_POSITIONS_ELEC[cathode][0]*width+x_offset)
        y_cathodes.append(ELECTRODE_POSITIONS_ELEC[cathode][1]*height+y_offset)
    x_anodes, y_anodes = np.atleast_1d(x_anodes, y_anodes)
    x_cathodes, y_cathodes = np.atleast_1d(x_cathodes, y_cathodes)
    ##
    # Plotting the Animation
    #numDataPoints = lhip.shape[0]-1
    numDataPoints = int(np.array(list(stim.iloc[selected_trial][stim.columns[-17:]].values)).shape[1])

    #LH = resample(kin.LHIPF.iloc[selected_trial], n_samples=numDataPoints, random_state=0)
    #LA = resample(kin.LAdd.iloc[selected_trial], n_samples=numDataPoints, random_state=0)
    #LK = resample(kin.LKNEX.iloc[selected_trial], n_samples=numDataPoints, random_state=0)

    #lhip, lknee, lank = to_coo(LH, LA,LK)

    lhip, lknee, lank = to_coo(kin.LHIPF.iloc[selected_trial], kin.LAdd.iloc[selected_trial],kin.LKNEX.iloc[selected_trial] )
    rhip, rknee, rank = to_coo(kin.RHIPF.iloc[selected_trial], kin.RAdd.iloc[selected_trial],kin.RKNEX.iloc[selected_trial] )
    duration =  int(1000*lhip.shape[0]/fskin)


    fig1,ax1 = plt.subplots(2,2)

    x = np.linspace(0, duration-1,lhip.shape[0])

    ######################### can be removed
    ax1[0][0].plot(x,kin.LHIPF.iloc[selected_trial],'x')
    ax1[0][1].plot(x,kin.LKNEX.iloc[selected_trial],'x')
    ax1[1][0].plot(x,kin.RHIPF.iloc[selected_trial],'x')
    ax1[1][1].plot(x,kin.RKNEX.iloc[selected_trial],'x')
    jointLHfunc = interpolate.interp1d(x, kin.LHIPF.iloc[selected_trial])
    jointLKfunc = interpolate.interp1d(x, kin.LKNEX.iloc[selected_trial])
    jointRHfunc = interpolate.interp1d(x, kin.RHIPF.iloc[selected_trial])
    jointRKfunc = interpolate.interp1d(x, kin.RKNEX.iloc[selected_trial])



    ax1[0][0].plot(np.linspace(0, duration-1,numDataPoints),jointLHfunc(np.linspace(0, duration-1,numDataPoints)).T)
    ax1[0][1].plot(np.linspace(0, duration-1,numDataPoints),jointLKfunc(np.linspace(0, duration-1,numDataPoints)).T)
    ax1[1][0].plot(np.linspace(0, duration-1,numDataPoints),jointRHfunc(np.linspace(0, duration-1,numDataPoints)).T)
    ax1[1][1].plot(np.linspace(0, duration-1,numDataPoints),jointRKfunc(np.linspace(0, duration-1,numDataPoints)).T)
    fig1.savefig('hello.png')
    ##############################33

    lhipfunc = interpolate.interp1d(x, lhip.T)
    lkneefunc = interpolate.interp1d(x, lknee.T)
    lankfunc = interpolate.interp1d(x, lank.T)
    rhipfunc = interpolate.interp1d(x, rhip.T)
    rkneefunc = interpolate.interp1d(x, rknee.T)
    rankfunc = interpolate.interp1d(x, rank.T)

    actual_time = np.linspace(0, duration-1,numDataPoints)
    #lhip = resample(lhip, n_samples=numDataPoints, random_state=0)
    #lknee = resample(lknee, n_samples=numDataPoints, random_state=0)
    #lank = resample(lank, n_samples=numDataPoints, random_state=0)
    lhip = lhipfunc(actual_time).T
    lank = lankfunc(actual_time).T
    lknee = lkneefunc(actual_time).T
    rhip = rhipfunc(actual_time).T
    rank = rankfunc(actual_time).T
    rknee = rkneefunc(actual_time).T


    window_width = 200
    shift = int(window_width/2)
    emg_trial1 = emg.iloc[selected_trial]
    #trial = data.filter(np.array(list(emg_trial1[MUSCLES].values)).T, fs=fsemg, lowcut=30, highcut=200)
    pooh= np.array(list(emg_trial1[MUSCLES].values)).T
    #emg_trial = resample(np.array(list(emg_trial1[MUSCLES].values)).T, n_samples=numDataPoints, random_state=0)
    emgfunc = interpolate.interp1d(np.linspace(0, duration-1,pooh.shape[0]), pooh.T)
    emg_trial = emgfunc(actual_time).T
    #emg_trial = resample(trial, n_samples=numDataPoints, random_state=0)

    #ticks = np.linspace(-shift, emg_trial.shape[0]-1+shift,int((emg_trial.shape[0]+window_width)/5)+1)
    #ticks = np.linspace(-shift, duration-1+shift,int((numDataPoints+window_width)/10)+1)
    ticks = np.arange(-50,duration-1+shift,50)

    lab = ['']
    for i in ticks[1:] :
        lab = np.append(lab, str(int(i)))
    #time = np.linspace(-shift, duration-1+shift,numDataPoints+window_width)
    #time = np.linspace(-shift, emg_trial.shape[0]-1+shift,emg_trial.shape[0]+window_width)
    arr = np.empty((int(window_width/2),emg_trial.shape[1]))
    arr[:] = 0#np.NaN

    #stim_duration = 397#emg_trial.shape[0]
    #stim_arrays = data.create(test_stim_features, stim_duration, fs=229)#data.FS)
    #stim = stim_arrays[selected_trial,0,:,:]
    stimtrial = np.array(list(stim.iloc[selected_trial][stim.columns[-17:]].values)).T
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.linspace(0,10,stimtrial.shape[0]),stimtrial)
    #ax[0].set_facecolor('red')
    #ax[0].plot(np.linspace(0,10,pooh.shape[0]),pooh-70)
    ax[0].plot(np.linspace(0,10,lknee.shape[0]),lknee[:,1:])
    #stimtrial = resample(stimtrial, n_samples=numDataPoints, random_state=0)
    stimfunc = interpolate.interp1d(np.linspace(0, duration-1,stimtrial.shape[0]), stimtrial.T)
    stimtrial = stimfunc(actual_time).T
    ##
    cathode_index = np.argmax(np.max(stimtrial, axis=0))
    peaks = find_peaks(stimtrial[:,cathode_index])

    begin = peaks[0][0] - 20
    ##
    ax[1].plot(np.linspace(0,10,stimtrial.shape[0]),15*stimtrial)
    ax[1].plot(np.linspace(0,10,emg_trial.shape[0]),emg_trial-70)
    plt.savefig('ooo.png')
    arr1 = np.zeros((int(window_width/2),17))
    arr2 = np.ones((int(window_width/2),3))
    arr3 = np.ones((int(window_width/2),1))
    ##
    LH = np.array([list(jointLHfunc(np.linspace(0, duration-1,numDataPoints)))]).T
    LK = np.array([list(jointLKfunc(np.linspace(0, duration-1,numDataPoints)))]).T
    LH = np.concatenate(( arr3*LH[0], LH[begin:],arr3*LH[-1] ), axis=0)
    LK = np.concatenate(( arr3*LK[0], LK[begin:],arr3*LK[-1] ), axis=0)
    if LEGS2 :
        RH = np.array([list(jointRHfunc(np.linspace(0, duration-1,numDataPoints)))]).T
        RK = np.array([list(jointRKfunc(np.linspace(0, duration-1,numDataPoints)))]).T
        RH = np.concatenate(( arr3*RH[0], RH[begin:],arr3*RH[-1] ), axis=0)
        RK = np.concatenate(( arr3*RK[0], RK[begin:],arr3*RK[-1] ), axis=0)
    print(LH.shape , LK.shape)
    ##
    exp_stim = np.concatenate((arr1, stimtrial[begin:],arr1), axis=0)
    trial =  np.concatenate((arr, emg_trial[begin:],arr), axis=0) # add nan before and after emg_trial and time

    lhip = np.concatenate(( arr2*lhip[0], lhip[begin:],arr2*lhip[-1] ), axis=0)
    lank = np.concatenate(( arr2*lank[0], lank[begin:],arr2*lank[-1] ), axis=0)
    lknee = np.concatenate((  arr2*lknee[0], lknee[begin:],arr2*lknee[-1] ), axis=0)
    rhip = np.concatenate((  arr2*rhip[0], rhip[begin:],arr2*rhip[-1] ), axis=0)
    rank = np.concatenate((  arr2*rank[0], rank[begin:],arr2*rank[-1] ), axis=0)
    rknee =  np.concatenate((  arr2*rknee[0], rknee[begin:],arr2*rknee[-1] ), axis=0)

    numDataPoints = exp_stim.shape[0]-1
    time = np.linspace(-1000*shift/stimfs, duration-1+shift-1000*begin/stimfs,numDataPoints+window_width)
    #trial = data.filter(trial, fs=numDataPoints/(duration/1000), lowcut=15, highcut=180)
    #pred_trial =  np.concatenate((arr, pred_trial,arr), axis=0)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.linspace(0,10,exp_stim.shape[0]),exp_stim)
    ax[1].plot(np.linspace(0,10,trial.shape[0]),trial)
    ax[0].plot(np.linspace(0,10,lknee.shape[0]),lknee[:,1:])
    #ax[0].set_facecolor('red')
    plt.savefig('aaa.png')



    ###################################################


    #a =  np.vstack((np.array([list(jointLHfunc(np.linspace(0, duration-1,numDataPoints)))]) ,  np.array([list(jointLKfunc(np.linspace(0, duration-1,numDataPoints)))]) ))
    a = np.vstack((LH.T, LK.T)).T
    if LEGS2 :
        a = np.vstack((LH.T, LK.T, RH.T , RK.T)).T

    return np.expand_dims(a, axis=0) , np.expand_dims(exp_stim, axis=0) , np.expand_dims(trial, axis=0)

wanted_trials = [7,41, 480, 502]#[7 , 40, 360, 155,486,505, 41, 163, 480,502, 481,499]
legs , stims_ , trial_ = get_data(wanted_trials[0], LEGS2 = True)
for nb in wanted_trials[1:]:
    leg , stim___ , trial = get_data(nb, LEGS2 = True)
    legs = np.concatenate((legs, leg), axis=0)
    stims_ = np.concatenate((stims_, stim___), axis=0)
    trial_ = np.concatenate((trial_, trial), axis=0)

print('Kinematics shape : ' , legs.shape)
with open('leftlegs40pluss.npy', 'wb') as f:
    np.save(f, legs,allow_pickle=False)

print('Stim shape : ', stims_.shape)
with open('stimlegs40pluss.npy', 'wb') as f:
    np.save(f, stims_, allow_pickle=False)

print('EMGs shape : ', trial_.shape)
with open('emgs40pluss.npy', 'wb') as f:
    np.save(f,trial_, allow_pickle=False)
