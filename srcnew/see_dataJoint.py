#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import pandas as pd
import _pickle as cPickle
import torch
import matplotlib.pyplot as plt
import data
from data import stim
from data.emg import realign_stim, realign_stimUnit, rolling
import joint_to_coo
import rnn
import utils as utils
import tests.params_files
from importlib import import_module, __import__
from sklearn.utils import resample
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from utils import plot_electrode_activation
import os
import matplotlib
import matplotlib.lines as mlines
import matplotlib.axes
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import List, Tuple
from data import PATH, N_ELECTRODES, MUSCLES
import scipy.io
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# Loading dataset
emg = cPickle.load(open("emgMay21.pkl", "rb"))
kin = cPickle.load(open("kinMay21.pkl", "rb"))
cut = cPickle.load(open("cutMay21.pkl", "rb"))

# Prepare convenient Dataframe
stimcol = list(kin.columns[:-6])
for i in range(17):
    name = "elec" + str(i + 1)
    stimcol.append(name)
new_kin = pd.DataFrame(columns=kin.columns)
new_stim = pd.DataFrame(columns=stimcol)
new_emg = pd.DataFrame(columns=emg.columns)

# Remove the 13Hz recordings
index_13 = emg.index[emg["Frequency"] == 13].tolist()
emg = emg.drop(index=index_13)
kin = kin.drop(index=index_13)
cut = cut.drop(index=index_13)

emg[["timestamp_start","timestamp_stop","Frequency","Amplitude","PulseWidth","Pulses"]] = emg[["timestamp_start","timestamp_stop","Frequency","Amplitude","PulseWidth","Pulses"]].astype(int)
emg_data = np.expand_dims(np.array(list(emg[emg.columns[-14:]].iloc[0, :].values))[:, :1763], axis=0)

for i in range(1, emg.shape[0]):
    emg_data = np.concatenate((emg_data,np.expand_dims(np.array(list(emg[emg.columns[-14:]].iloc[i, :].values))[:, :1763],axis=0)),axis=0)
emg_plot = emg_data

#####################################
# Organize stim, kin and EMG dataframes
dfkin = pd.DataFrame(columns=["RHIPF", "RAdd", "RKNEX", "LHIPF", "LAdd", "LKNEX"])

fsemg = 1259.25925925926
duration_ms = int(np.floor(1000 * len(cut.iloc[0][0]) / fsemg))
fskin = 148.148
stimfs = 10 * fskin

# Stim dataframe
stimold = data.create(
    emg[["Frequency", "Pulses", "PulseWidth", "Amplitude", "Anodes", "Cathodes"]],
    duration_ms,
    stimfs,
)

for i in range(emg.shape[0]):  # For each trial
    selected_trial = i
    rowkin = list(kin.iloc[selected_trial, :-6].values)
    rowstim = list(kin.iloc[selected_trial, :-6].values)
    rowemg = list(emg.iloc[selected_trial, :-14].values)

    emg_trial = emg_plot[selected_trial, :, :]

    tot_dur = emg_trial.shape[1] / fsemg
    stim_on = cut.iloc[i][0]
    duration_ms = int(np.floor(1000 * len(stim_on) / fsemg))
    initial_seg_dur = stim_on[0] / fsemg
    final_seg_dur = (emg_trial.shape[1] - stim_on[-1]) / fsemg
    # add segments to joints
    before = np.ones(int(initial_seg_dur * fskin))

    # Kin dataframe
    joints = []
    for joint in ["RHIPF", "RAdd", "RKNEX", "LHIPF", "LAdd", "LKNEX"]:

        current = kin.iloc[selected_trial][joint]
        joints.append(np.concatenate((np.multiply(before, kin.iloc[selected_trial][joint][0]), current),axis=0))
    dfkin.loc[len(dfkin.index)] = joints

    # Compute joint coordinates
    rhip, rknee, rank = joint_to_coo.to_coo(
        dfkin.iloc[selected_trial].RHIPF,
        dfkin.iloc[selected_trial].RAdd,
        dfkin.iloc[selected_trial].RKNEX,
    )
    lhip, lknee, lank = joint_to_coo.to_coo(
        dfkin.iloc[selected_trial].LHIPF,
        dfkin.iloc[selected_trial].LAdd,
        dfkin.iloc[selected_trial].LKNEX,
    )
    emg_trial = emg_trial.T

    ##########

    # Realign stim
    arr_pre = np.empty((int(initial_seg_dur * stimfs), 17))
    arr_pre[:] = 0
    arr_post = np.empty((int(final_seg_dur * stimfs), 17))
    arr_post[:] = 0

    stim = np.concatenate((arr_pre, stimold[selected_trial, 0, :, :], arr_post), axis=0)
    stim, delay = rolling(stim, int(stim_on[0] * stimfs / fsemg))
    stim = stim[: -int(final_seg_dur * stimfs), :]
    trigger = (arr_pre.shape[0] + delay) * 1000 / stimfs  # ms
    time_forstim = np.linspace(0, (lhip[:, 2].shape[0] * 1000 / fskin) - 1, stim.shape[0])
    time = np.linspace(0, (lhip[:, 2].shape[0] * 1000 / fskin) - 1, lhip[:, 2].shape[0])

    emg_trial = emg_trial[: stim_on[-1], :]
    timeemg = np.linspace(0, emg_trial.shape[0] - 1, emg_trial.shape[0])

    # PLots
    fig, ax = plt.subplots(4, 2, figsize=(13, 10), gridspec_kw={"height_ratios": [1, 4, 2, 2]})
    ax[0, 0].set_title("LEFT", fontweight="bold")
    ax[0, 1].set_title("RIGHT", fontweight="bold")

    ax[0][0].set_ylim([-4, 3])
    ax[0][1].set_yticks([])
    ax[0][1].set_xticks([])
    ax[0][0].set_xticks([])
    ax[0][0].set_ylabel("Stim [mA]", fontsize="8")
    ax[0][1].set_ylim([-5, 3])
    ax[0][0].set_frame_on(False)
    ax[0][1].set_frame_on(False)
    # Plot the stim
    ax[0][0].plot(time_forstim, stim, color="#fa525b")
    ax[0][0].axvline(x=time_forstim[arr_pre.shape[0] + delay], color="red", linestyle="--", alpha=0.3)
    ax[0][1].plot(time_forstim, stim, color="#fa525b")
    ax[0][1].axvline(x=time_forstim[arr_pre.shape[0] + delay], color="red", linestyle="--", alpha=0.3)

    for i in range(17):
        rowstim.append(stim[:, i])
    ax[0][1].set_title(str(emg.iloc[selected_trial : selected_trial + 1].Pulses.values[0])+ " pulses      ",fontsize=12,loc="right")

    for side in [0, 1]:
        ax[1][side].axvline(x=trigger * fsemg / 1000, color="red", alpha=0.2, linestyle="--")
        for r in range(7):
            predicted_line = ax[1][side].plot(
                timeemg,
                emg_trial[:, r + 7 * (side)] - r * 230,
                linewidth=1,
                color="lightgrey",
            )
            rowemg.append(emg_trial[:, r + 7 * (side)])

            ax[1][side].set_frame_on(False)
            ax[1][side].set_ylim([-1500, 130])
            ax[1][side].set_xticks([])
            labs = ["100","0","-100","100","0","-100","100","0","-100","100","0","-100","100","0","-100","100","0","-100","100","0","-100"]
            ax[1][0].yaxis.set_minor_locator(ticker.FixedLocator([-1480,-1380,-1280,-1250,-1150,-1050,-1020,-920,-820,-790,-690,-590,-560,-460,-360,-330,-230,-130,-100,0,100]))
            ax[1][0].yaxis.set_minor_formatter(ticker.FixedFormatter(labs))
            ax[1][0].tick_params(which="minor", length=3, labelcolor="silver", labelsize="x-small")
            ax[1][side].tick_params("x", labelbottom=True, labelsize="10")
    loc = [-1380, -1150, -920, -690, -460, -230, 0]
    labels = ["Sol", "TA", "MG", "ST", "VLat", "RF", "Add"]
    for i in range(7):
        ax[1][0].text(-13, loc[i] - 20, labels[i], fontsize="large")
    ax[1][0].set_title("Muscles EMGs", fontsize=12, loc="left")
    ax[1][1].set_yticks([])
    ax[1][0].set_yticks([])
    ax[0][1].set_yticks([])
    ax[1][0].set_ylabel("Normalized EMG", fontsize="8")

    # plot the EMG
    letter = ["L", "R"]
    for side in [0, 1]:
        # plot z coordinate
        ax[2][0].set_title("Joint angles", fontsize=12, loc="left")
        ax[2][side].set_frame_on(False)
        ax[2][side].plot(
            time,
            dfkin[letter[side] + "Add"].iloc[selected_trial],
            color="darkorange",
            linewidth=1,
            label=letter[side] + "Add",
        )
        ax[2][side].plot(
            time,
            dfkin[letter[side] + "KNEX"].iloc[selected_trial],
            color="orchid",
            linewidth=1,
            label=letter[side] + "KNEX",
        )
        ax[2][side].plot(
            time,
            dfkin[letter[side] + "HIPF"].iloc[selected_trial],
            color="mediumaquamarine",
            linewidth=1,
            label=letter[side] + "HIPF",
        )
        ax[2][side].axvline(x=trigger, color="red", linestyle="--", alpha=0.2)

        rowkin.append(dfkin[letter[side] + "HIPF"].iloc[selected_trial])
        rowkin.append(dfkin[letter[side] + "Add"].iloc[selected_trial])
        rowkin.append(dfkin[letter[side] + "KNEX"].iloc[selected_trial])
        ax[2][side].tick_params(colors="silver")
        ax[2][1].set_ylim([-80, 80])
        ax[2][0].set_ylim([-80, 80])
        ax[2][1].set_yticks([])
        ax[2][1].set_xticks([])
        ax[2][0].set_xticks([])
    ax[2][0].set_ylabel("Angle [degree]", fontsize="8")
    ax[2][1].legend()

    ax[3][0].set_title("Z-Coordinates", fontsize=12, loc="left")
    ax[3][0].set_frame_on(False)
    ax[3][1].set_frame_on(False)

    ax[3][0].plot(time, lhip[:, 2], color="tomato", linewidth=1, label="Hip")
    ax[3][0].plot(time, lknee[:, 2], color="lightgreen", linewidth=1, label="Knee")
    ax[3][0].plot(time, lank[:, 2], color="lightskyblue", linewidth=1, label="Ankle")
    ax[3][0].axvline(x=trigger, color="red", linestyle="--", alpha=0.2)

    ax[3][1].plot(time, rhip[:, 2], color="tomato", linewidth=1, label="Hip")
    ax[3][1].plot(time, rknee[:, 2], color="lightgreen", linewidth=1, label="Knee")
    ax[3][1].plot(time, rank[:, 2], color="lightskyblue", linewidth=1, label="Ankle")
    ax[3][1].axvline(x=trigger, color="red", linestyle="--", alpha=0.2)

    ax[3][side].set_xlabel("Time [ms]", fontsize="12")
    ax[3][side].tick_params(colors="silver")
    ax[3][0].set_xlabel("Time [ms]", fontsize="12")
    ax[3][1].tick_params(colors="silver")
    ax[3][1].set_ylim([-10, 40])
    ax[3][0].set_ylim([-10, 40])
    ax[3][1].set_yticks([])
    ax[3][0].set_ylabel("Z-Coordinate", fontsize="8")
    ax[3][1].legend()

    fig.suptitle("Freq: "+ str(emg.Frequency.iloc[selected_trial])+ "Hz  Amp : "+ str(emg.Amplitude.iloc[selected_trial])+ "mA  Cathodes : "+ str(emg.Cathodes.iloc[selected_trial])+ "  Anodes : "+ str(emg.Anodes.iloc[selected_trial]))
    new_emg.loc[len(new_emg)] = rowemg
    new_kin.loc[len(new_kin)] = rowkin
    new_stim.loc[len(new_stim)] = rowstim

    plt.tight_layout()
    plt.show()
    f = ("C:/Users/yes/Documents/GitHub/little_RNN/src/observedataJoint0/trial"+ str(selected_trial)+ ".png")
    fig.savefig(f)

# save in _picklefile
new_emg.to_pickle("processednot_norm_emg.pkl")
new_kin.to_pickle("processed_kin.pkl")
new_stim.to_pickle("processed_stim.pkl")
