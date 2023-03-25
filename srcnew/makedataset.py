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
import os
from utils import ELECTRODE_POSITIONS_ELEC
import matplotlib
import matplotlib.lines as mlines
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
# selected_trial = 54
# selected_trial = emg.loc[((emg['Cathodes'].astype(str) == '[10]') &(emg['Anodes'].astype(str) == '[16]')  & (emg['Frequency'] == 120)& (emg['Amplitude'] == 3.5))].index.values[0]
###################################################

# Loading data

emgold = cPickle.load(open("processednot_norm_emg1.pkl", "rb"))
kin = cPickle.load(open("processed_kin1.pkl", "rb"))
stim = cPickle.load(open("processed_stim1.pkl", "rb"))

fsemg = 1259.25925925926
fskin = 148.148
stimfs = 10 * fskin

# Useful functions


def normalize(emg_df, cols_to_cropreplace, emgfs):
    nbpt_remove = int(emgfs * 0.4)
    for muscle in cols_to_cropreplace:
        rows = []
        max_ = []
        for row in emg_df[muscle].values:
            rows.append(100 * row[:])
            max_.append(np.max(np.abs(row[nbpt_remove:])))
        del emg_df[muscle]
        emg_df[muscle] = rows / np.max(max_)
    return emg_df


def get_data(selected_trial, LEGS2=False):  # LEGS2 = whether of not you need data of right leg
    emg = normalize(emgold, list(emgold.columns[-14:]), fsemg)
    # can store it in pickle files
    # emg.iloc[selected_trial].to_pickle("goodEMGoverfit.pkl")
    # kin.iloc[selected_trial].to_pickle("goodKINoverfit.pkl")
    # stim.iloc[selected_trial].to_pickle("goodSTIMoverfit.pkl")
    ########s

    def to_coo(HIPF, ADD, KNEX):
        KNEX = KNEX + -70
        HIPF = HIPF + 5
        l_thigh = 23
        l_shank = 25
        coo_hip = []
        coo_knee = []
        coo_ankle = []
        for t in range(len(HIPF)):
            knee_x = l_thigh * np.cos(np.radians(HIPF[t])) * np.cos(np.radians(ADD[t]))
            knee_y = l_thigh * np.cos(np.radians(HIPF[t])) * np.sin(np.radians(ADD[t]))
            knee_z = l_thigh * np.sin(np.radians(HIPF[t]))
            l_hiptoankle = np.sqrt(l_thigh**2+ l_shank**2- 2 * l_thigh * l_shank * np.cos(np.radians(180 - KNEX[t])))
            theta = math.degrees(math.acos((l_thigh**2 + l_hiptoankle**2 - l_shank**2)/ (2 * l_thigh * l_hiptoankle)))
            alpha = HIPF[t] - theta
            ankle_x = (l_hiptoankle * np.cos(np.radians(alpha)) * np.cos(np.radians(ADD[t])))
            ankle_y = (l_hiptoankle * np.cos(np.radians(alpha)) * np.sin(np.radians(ADD[t])))
            ankle_z = l_hiptoankle * np.sin(np.radians(alpha))
            coo_hip.append(np.array([0, 0, 0]))
            coo_knee.append(np.array([knee_x, knee_y, knee_z]))
            coo_ankle.append(np.array([ankle_x, ankle_y, ankle_z]))
        return np.array(coo_hip), np.array(coo_knee), np.array(coo_ankle)

    ###################################################
    # SETTINGS

    with open(os.path.abspath("../images/electrode.png"), "rb") as electrode_file:
        electrode_im = plt.imread(electrode_file)
    height, width = electrode_im.shape[0], electrode_im.shape[1]
    x_offset = 4
    y_offset = 90
    x_anodes, y_anodes = [], []
    for anode in kin["Anodes"].iloc[selected_trial]:
        x_anodes.append(ELECTRODE_POSITIONS_ELEC[anode][0] * width + x_offset)
        y_anodes.append(ELECTRODE_POSITIONS_ELEC[anode][1] * height + y_offset)
    x_cathodes, y_cathodes = [], []
    for cathode in kin["Cathodes"].iloc[selected_trial]:
        x_cathodes.append(ELECTRODE_POSITIONS_ELEC[cathode][0] * width + x_offset)
        y_cathodes.append(ELECTRODE_POSITIONS_ELEC[cathode][1] * height + y_offset)
    x_anodes, y_anodes = np.atleast_1d(x_anodes, y_anodes)
    x_cathodes, y_cathodes = np.atleast_1d(x_cathodes, y_cathodes)

    # Plotting the Animation

    numDataPoints = int(np.array(list(stim.iloc[selected_trial][stim.columns[-17:]].values)).shape[1])

    lhip, lknee, lank = to_coo(
        kin.LHIPF.iloc[selected_trial],
        kin.LAdd.iloc[selected_trial],
        kin.LKNEX.iloc[selected_trial],
    )
    rhip, rknee, rank = to_coo(
        kin.RHIPF.iloc[selected_trial],
        kin.RAdd.iloc[selected_trial],
        kin.RKNEX.iloc[selected_trial],
    )
    duration = int(1000 * lhip.shape[0] / fskin)

    fig1, ax1 = plt.subplots(2, 2)

    x = np.linspace(0, duration - 1, lhip.shape[0])

    ax1[0][0].plot(x, kin.LHIPF.iloc[selected_trial], "x")
    ax1[0][1].plot(x, kin.LKNEX.iloc[selected_trial], "x")
    ax1[1][0].plot(x, kin.RHIPF.iloc[selected_trial], "x")
    ax1[1][1].plot(x, kin.RKNEX.iloc[selected_trial], "x")
    jointLHfunc = interpolate.interp1d(x, kin.LHIPF.iloc[selected_trial])
    jointLKfunc = interpolate.interp1d(x, kin.LKNEX.iloc[selected_trial])
    jointRHfunc = interpolate.interp1d(x, kin.RHIPF.iloc[selected_trial])
    jointRKfunc = interpolate.interp1d(x, kin.RKNEX.iloc[selected_trial])

    ax1[0][0].plot(
        np.linspace(0, duration - 1, numDataPoints),
        jointLHfunc(np.linspace(0, duration - 1, numDataPoints)).T,
    )
    ax1[0][1].plot(
        np.linspace(0, duration - 1, numDataPoints),
        jointLKfunc(np.linspace(0, duration - 1, numDataPoints)).T,
    )
    ax1[1][0].plot(
        np.linspace(0, duration - 1, numDataPoints),
        jointRHfunc(np.linspace(0, duration - 1, numDataPoints)).T,
    )
    ax1[1][1].plot(
        np.linspace(0, duration - 1, numDataPoints),
        jointRKfunc(np.linspace(0, duration - 1, numDataPoints)).T,
    )
    fig1.savefig("kin.png")
    ##############################33

    lhipfunc = interpolate.interp1d(x, lhip.T)
    lkneefunc = interpolate.interp1d(x, lknee.T)
    lankfunc = interpolate.interp1d(x, lank.T)
    rhipfunc = interpolate.interp1d(x, rhip.T)
    rkneefunc = interpolate.interp1d(x, rknee.T)
    rankfunc = interpolate.interp1d(x, rank.T)

    actual_time = np.linspace(0, duration - 1, numDataPoints)
    lhip = lhipfunc(actual_time).T
    lank = lankfunc(actual_time).T
    lknee = lkneefunc(actual_time).T
    rhip = rhipfunc(actual_time).T
    rank = rankfunc(actual_time).T
    rknee = rkneefunc(actual_time).T

    window_width = 200
    shift = int(window_width / 2)
    emg_trial1 = emg.iloc[selected_trial]
    pooh = np.array(list(emg_trial1[MUSCLES].values)).T
    emgfunc = interpolate.interp1d(np.linspace(0, duration - 1, pooh.shape[0]), pooh.T)
    emg_trial = emgfunc(actual_time).T
    ticks = np.arange(-50, duration - 1 + shift, 50)

    lab = [""]
    for i in ticks[1:]:
        lab = np.append(lab, str(int(i)))
    arr = np.empty((int(window_width / 2), emg_trial.shape[1]))
    arr[:] = 0
    stimtrial = np.array(list(stim.iloc[selected_trial][stim.columns[-17:]].values)).T
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.linspace(0, 10, stimtrial.shape[0]), stimtrial)
    ax[0].plot(np.linspace(0, 10, lknee.shape[0]), lknee[:, 1:])
    stimfunc = interpolate.interp1d(np.linspace(0, duration - 1, stimtrial.shape[0]), stimtrial.T)
    stimtrial = stimfunc(actual_time).T

    cathode_index = np.argmax(np.max(stimtrial, axis=0))
    peaks = find_peaks(stimtrial[:, cathode_index])

    begin = peaks[0][0] - 20

    ax[1].plot(np.linspace(0, 10, stimtrial.shape[0]), 15 * stimtrial)
    ax[1].plot(np.linspace(0, 10, emg_trial.shape[0]), emg_trial - 70)
    plt.savefig("emg.png")
    arr1 = np.zeros((int(window_width / 2), 17))
    arr2 = np.ones((int(window_width / 2), 3))
    arr3 = np.ones((int(window_width / 2), 1))

    LH = np.array([list(jointLHfunc(np.linspace(0, duration - 1, numDataPoints)))]).T
    LK = np.array([list(jointLKfunc(np.linspace(0, duration - 1, numDataPoints)))]).T
    LH = np.concatenate((arr3 * LH[0], LH[begin:], arr3 * LH[-1]), axis=0)
    LK = np.concatenate((arr3 * LK[0], LK[begin:], arr3 * LK[-1]), axis=0)

    # If want data of the right leg as well
    if LEGS2:
        RH = np.array([list(jointRHfunc(np.linspace(0, duration - 1, numDataPoints)))]).T
        RK = np.array([list(jointRKfunc(np.linspace(0, duration - 1, numDataPoints)))]).T
        RH = np.concatenate((arr3 * RH[0], RH[begin:], arr3 * RH[-1]), axis=0)
        RK = np.concatenate((arr3 * RK[0], RK[begin:], arr3 * RK[-1]), axis=0)
    exp_stim = np.concatenate((arr1, stimtrial[begin:], arr1), axis=0)
    trial = np.concatenate((arr, emg_trial[begin:], arr), axis=0)  # add nan before and after emg_trial and time

    lhip = np.concatenate((arr2 * lhip[0], lhip[begin:], arr2 * lhip[-1]), axis=0)
    lank = np.concatenate((arr2 * lank[0], lank[begin:], arr2 * lank[-1]), axis=0)
    lknee = np.concatenate((arr2 * lknee[0], lknee[begin:], arr2 * lknee[-1]), axis=0)
    rhip = np.concatenate((arr2 * rhip[0], rhip[begin:], arr2 * rhip[-1]), axis=0)
    rank = np.concatenate((arr2 * rank[0], rank[begin:], arr2 * rank[-1]), axis=0)
    rknee = np.concatenate((arr2 * rknee[0], rknee[begin:], arr2 * rknee[-1]), axis=0)

    numDataPoints = exp_stim.shape[0] - 1
    time = np.linspace(-1000 * shift / stimfs,duration - 1 + shift - 1000 * begin / stimfs,numDataPoints + window_width,)
    ##
    a = np.vstack((LH.T, LK.T)).T

    if LEGS2:
        a = np.vstack((LH.T, LK.T, RH.T, RK.T)).T

    return (np.expand_dims(a, axis=0),np.expand_dims(exp_stim, axis=0),np.expand_dims(trial, axis=0),)


# Insert the trials' numbers you want to include in the dataset
wanted_trials = [7,41,480,502]
legs, stims_, trial_ = get_data(wanted_trials[0], LEGS2=True)
for nb in wanted_trials[1:]:
    leg, stim___, trial = get_data(nb, LEGS2=True)
    legs = np.concatenate((legs, leg), axis=0)
    stims_ = np.concatenate((stims_, stim___), axis=0)
    trial_ = np.concatenate((trial_, trial), axis=0)

print("Kinematics shape : ", legs.shape)
with open("leftlegs40pluss.npy", "wb") as f:
    np.save(f, legs, allow_pickle=False)

print("Stim shape : ", stims_.shape)
with open("stimlegs40pluss.npy", "wb") as f:
    np.save(f, stims_, allow_pickle=False)

print("EMGs shape : ", trial_.shape)
with open("emgs40pluss.npy", "wb") as f:
    np.save(f, trial_, allow_pickle=False)
