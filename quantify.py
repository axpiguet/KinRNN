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
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from utils import plot_electrode_activation
import os
import matplotlib.lines as mlines
import matplotlib.axes
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import List, Tuple
from data import PATH, N_ELECTRODES, MUSCLES
from utils import ELECTRODE_POSITIONS_ELEC
import scipy.io
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import seaborn as sns
import matplotlib.colors

# Loading dataset
emgold = cPickle.load(open("processednot_norm_emg1.pkl", "rb"))
kinold = cPickle.load(open("processed_kin1.pkl", "rb"))
stimold = cPickle.load(open("processed_stim1.pkl", "rb"))

fsemg = 1259.25925925926
fskin = 148.148
stimfs = 10 * fskin


def normalize_cropped(emg_df, cols_to_cropreplace, emgfs):
    nbpt_remove = int(emgfs * 0.4)
    for muscle in cols_to_cropreplace:
        rows = []
        max_ = []
        for row in emg_df[muscle].values:
            rows.append(100 * row[nbpt_remove:])
            max_.append(np.max(np.abs(row[nbpt_remove:])))
        del emg_df[muscle]
        emg_df[muscle] = rows / np.max(max_)
    return emg_df


def normalize(emg_df, cols_to_cropreplace, emgfs):
    for muscle in cols_to_cropreplace:
        rows = []
        max_ = []
        for row in emg_df[muscle].values:
            rows.append(100 * row[:])  # / np.max(np.abs(row[nbpt_remove:])))
            max_.append(np.max(np.abs(row[:])))
        del emg_df[muscle]
        emg_df[muscle] = rows / np.max(max_)
    return emg_df


def cropper(df, fs, cols_to_cropreplace):
    nbpt_remove = int(fs * 0.4)
    for col in cols_to_cropreplace:
        rows = []
        for row in df[col].values:
            rows.append(row[nbpt_remove:])
        del df[col]
        df[col] = rows
    return df


# Data processing
emg = normalize_cropped(emgold, list(emgold.columns[-14:]), fsemg)
kin = cropper(kinold, fskin, list(kinold.columns[-6:]))
stim = cropper(stimold, stimfs, list(stimold.columns[-17:]))

####### Quantify #######


def max_acti(df):
    for col in list(df.columns[-14:]):
        maxs = []
        for row in df[col].values:
            maxs.append(np.max(np.abs(row)))
        df["".join([col, "max"])] = maxs
    return df


def max_angle(df):
    for col in list(df.columns[-6:]):
        maxs = []
        for row in df[col].values:
            maxs.append(np.max(np.abs(row)))
        df["".join([col, "max"])] = maxs
    return df


kin = max_angle(kin)
emg = max_acti(emg)

####### Plot functions #######


def draw_per_Cath(emgdf, muscle):
    cathodes = emgold["Cathodes"].astype(str).unique()

    fig, axes = plt.subplots(
        ncols=4 + 1,
        nrows=int(np.ceil(len(cathodes) / 4)),
        figsize=(10, 10),
        gridspec_kw={"width_ratios": [4, 4, 4, 4, 1]},
    )
    gs = axes[0, 4].get_gridspec()
    for ax in axes[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, -1])

    sns.set(rc={"figure.figsize": (10, 20)})
    cmap_ = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "#fa525b"])
    for i in range(int(np.ceil(len(cathodes) / 4))):
        for j in range(4):
            axes[i][j].set_frame_on(False)
            axes[i][j].set_xlabel(" ")
            axes[i][j].set_ylabel(" ")
            axes[i][j].set_yticks([])
            axes[i][j].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            axes[i][j].set_xticks([])
    for i, cath in enumerate(cathodes):
        data = emg[emg["Cathodes"].astype(str) == cath]
        axes[i // 4, 0].get_shared_y_axes().join(axes[i // 4, 1], axes[i // 4, 2], axes[i // 4, 3])
        df2 = pd.crosstab(
            emg["Frequency"],
            emg["Amplitude"],
            values=data["".join([muscle, "max"])],
            aggfunc="mean",
            dropna=False ).fillna(0)
        sns.heatmap(
            df2,
            annot=False,
            cmap=cmap_,
            cbar_ax=axbig,
            ax=axes[i // 4, i % 4],
            vmin=0,
            vmax=100,
            mask=False,
            cbar_kws={"aspect": 100, "label": "Max " + muscle + " activation [%]"})

        axes[i // 4, i % 4].set_title("Cathode " + cath, fontsize="small", y=-0.15)
        axes[i // 4, i % 4].tick_params(axis="both", labelsize="x-small", colors="silver")
        axes[i // 4, i % 4].tick_params(axis="y", labelrotation=0)
        if not (i % 4 == 0):
            axes[i // 4, i % 4].set_yticks([])
        if not (i // 4 == 0):
            axes[i // 4, i % 4].set_xticks([])

    for i in range(int(np.ceil(len(cathodes) / 4))):
        for j in range(4):
            axes[i][j].set_frame_on(False)
            axes[i][j].set_xlabel(" ")
            axes[i][j].set_ylabel(" ")

    fig.suptitle("Amplitude [mA]", color="w", fontsize="large")
    fig.supylabel("Frequency [Hz]", color="w", fontsize="large")
    plt.tight_layout()
    f = (r"c://Users/axell/Desktop/Master_Thesis/little_RNN/src/Datainsight/BestCathodes/"+ muscle+ ".png")
    fig.savefig(f)


def draw_allconf(emgdf, muscle):
    confs = emgold["ElectrodeConftxt"].unique()

    fig, axes = plt.subplots(
        ncols=8,
        nrows=8,
        figsize=(10, 10),
        gridspec_kw={"width_ratios": [4, 4, 4, 4, 4, 4, 4, 1]})
    gs = axes[0, 7].get_gridspec()
    for ax in axes[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, -1])

    sns.set(rc={"figure.figsize": (10, 20)})
    cmap_ = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "#fa525b"])
    for i in range(8):
        for j in range(7):
            axes[i][j].set_frame_on(False)
            axes[i][j].set_xlabel(" ")
            axes[i][j].set_ylabel(" ")
            axes[i][j].set_yticks([])
            axes[i][j].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            axes[i][j].set_xticks([])
    for i, conf in enumerate(confs):
        data = emg[emg["ElectrodeConftxt"].astype(str) == conf]
        axes[i // 7, 0].get_shared_y_axes().join(
            axes[i // 7, 1],
            axes[i // 7, 2],
            axes[i // 7, 3],
            axes[i // 7, 4],
            axes[i // 7, 5],
            axes[i // 7, 6],
        )
        df2 = pd.crosstab(
            emg["Frequency"],
            emg["Amplitude"],
            values=data["".join([muscle, "max"])],
            aggfunc="mean",
            dropna=False).fillna(0)
        sns.heatmap(
            df2,
            annot=False,
            cmap=cmap_,
            cbar_ax=axbig,
            ax=axes[i // 7, i % 7],
            vmin=0,
            vmax=100,
            mask=False,
            cbar_kws={"aspect": 100, "label": "Max " + muscle + " activation [%]"})
        axes[i // 7, i % 7].set_title(conf, fontsize="x-small", y=-0.3)
        axes[i // 7, i % 7].tick_params(axis="both", labelsize="x-small", colors="silver")
        axes[i // 7, i % 7].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        axes[i // 7, i % 7].set_yticklabels(["20", "40", "60", "80", "100", "120"])
        axes[i // 7, i % 7].tick_params(axis="y", labelrotation=0)

        if not (i % 7 == 0):
            axes[i // 7, i % 7].set_yticks([])
        if not (i // 7 == 0):
            axes[i // 7, i % 7].set_xticks([])
    for i in range(8):
        for j in range(7):
            axes[i][j].set_frame_on(False)
            axes[i][j].set_xlabel(" ")
            axes[i][j].set_ylabel(" ")

    fig.suptitle("Amplitude [mA]", color="w", fontsize="large")
    fig.supylabel("Frequency [Hz]", color="w", fontsize="large")
    plt.tight_layout()
    f = (r"c://Users/axell/Desktop/Master_Thesis/little_RNN/src/Datainsight/Allconf/"+ muscle+ ".png")
    fig.savefig(f)

cathodeplot = {
    # horizontal x vertical
    0: (2, 1),  # elect 1
    1: (0, 1),  # elect 2 (0,-1)
    2: (8, 1),  # elect 3 #0.437
    3: (8, 0),  # elect 4 #0.3045
    4: (7, 0),  # elect 5
    5: (5, 0),  # elect 6
    6: (3, 0),  # elect 7
    7: (1, 0),  # elect 8
    8: (8, 2),  # elect 9
    9: (8, 3),  # elect 10
    10: (7, 2),  # elect 11
    11: (5, 2),  # elect 12
    12: (3, 2),  # elect 13
    13: (1, 2),  # elect 14
    14: (6, 1),  # elect 15
    15: (4, 1),  # elect 16
    16: (np.nan, np.nan)}


def draw_per_CathSPAT(emgdf, muscle):
    cathodes = emgold["Cathodes"].astype(str).unique()

    fig, axes = plt.subplots(
        ncols=4 + 1,
        nrows=9,
        figsize=(10, 10),
        gridspec_kw={"width_ratios": [4, 4, 4, 4, 1]})
    gs = axes[0, 4].get_gridspec()
    for ax in axes[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, -1])

    sns.set(rc={"figure.figsize": (10, 20)})
    cmap_ = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "#fa525b"])
    for i in range(9):
        for j in range(4):
            axes[i][j].set_frame_on(False)
            axes[i][j].set_xlabel(" ")
            axes[i][j].set_ylabel(" ")
            axes[i][j].set_yticks([])
            axes[i][j].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            axes[i][j].set_xticks([])
    for i, cath in enumerate(cathodes):
        data = emg[emg["Cathodes"].astype(str) == cath]
        cathode_nb = int(cath.replace("[", "").replace("]", ""))

        df2 = pd.crosstab(
            emg["Frequency"],
            emg["Amplitude"],
            values=data["".join([muscle, "max"])],
            aggfunc="mean",
            dropna=False).fillna(0)
        sns.heatmap(
            df2,
            annot=False,
            cmap=cmap_,
            cbar_ax=axbig,
            ax=axes[cathodeplot[cathode_nb]],
            vmin=0,
            vmax=100,
            mask=False,
            cbar_kws={"aspect": 100, "label": "Max " + muscle + " activation [%]"})
        axes[cathodeplot[cathode_nb]].set_title("Cathode " + cath, fontsize="small", y=-0.15)
        axes[cathodeplot[cathode_nb]].tick_params(axis="both", labelsize="x-small", colors="silver")
        axes[cathodeplot[cathode_nb]].tick_params(axis="y", labelrotation=0)
        if not (cathodeplot[cathode_nb][1] == 0):
            axes[cathodeplot[cathode_nb]].set_yticks([])
        if not (cathodeplot[cathode_nb][0] == 0):
            axes[cathodeplot[cathode_nb]].set_xticks([])

    for i in range(9):
        for j in range(4):
            axes[i][j].set_frame_on(False)
            axes[i][j].set_xlabel(" ")
            axes[i][j].set_ylabel(" ")

    fig.suptitle("Amplitude [mA]", color="w", fontsize="large")
    fig.supylabel("Frequency [Hz]", color="w", fontsize="large")
    plt.tight_layout()
    f = (r"c://Users/axell/Desktop/Master_Thesis/little_RNN/src/Datainsight/BestCathodesSPAT/"+ muscle+ ".png")
    fig.savefig(f)


def draw_per_CathAngle(emgdf, angle):
    cathodes = emgold["Cathodes"].astype(str).unique()

    fig, axes = plt.subplots(
        ncols=4 + 1,
        nrows=9,
        figsize=(10, 10),
        gridspec_kw={"width_ratios": [4, 4, 4, 4, 1]}
    gs = axes[0, 4].get_gridspec()
    for ax in axes[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, -1])

    sns.set(rc={"figure.figsize": (10, 20)})
    cmap_ = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "dodgerblue"])
    for i in range(9):
        for j in range(4):
            axes[i][j].set_frame_on(False)
            axes[i][j].set_xlabel(" ")
            axes[i][j].set_ylabel(" ")
            axes[i][j].set_yticks([])
            axes[i][j].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            axes[i][j].set_xticks([])
    for i, cath in enumerate(cathodes):
        data = emgdf[emgdf["Cathodes"].astype(str) == cath]
        cathode_nb = int(cath.replace("[", "").replace("]", ""))
        df2 = pd.crosstab(
            emgdf["Frequency"],
            emgdf["Amplitude"],
            values=data[angle],
            aggfunc="mean",
            dropna=False).fillna(0)
        sns.heatmap(
            df2,
            annot=False,
            cmap=cmap_,
            cbar_ax=axbig,
            ax=axes[cathodeplot[cathode_nb]],
            vmin=0,
            vmax=100,
            mask=False,
            cbar_kws={"aspect": 100, "label": "Max " + angle + " [degrees]"})
        axes[cathodeplot[cathode_nb]].set_title("Cathode " + cath, fontsize="small", y=-0.15)
        axes[cathodeplot[cathode_nb]].tick_params(axis="both", labelsize="x-small", colors="silver")
        axes[cathodeplot[cathode_nb]].tick_params(axis="y", labelrotation=0)
        if not (cathodeplot[cathode_nb][1] == 0):
            axes[cathodeplot[cathode_nb]].set_yticks([])
        if not (cathodeplot[cathode_nb][0] == 0):
            axes[cathodeplot[cathode_nb]].set_xticks([])

    for i in range(9):
        for j in range(4):
            axes[i][j].set_frame_on(False)
            axes[i][j].set_xlabel(" ")
            axes[i][j].set_ylabel(" ")

    fig.suptitle("Amplitude [mA]", color="w", fontsize="large")
    fig.supylabel("Frequency [Hz]", color="w", fontsize="large")
    plt.tight_layout()
    f = (r"c://Users/axell/Desktop/Master_Thesis/little_RNN/src/Datainsight/BestCathodesSPATangle/"+ angle+ ".png")
    fig.savefig(f)


def draw_per_1conf(emgdf, conf, angles, muscles):
    cathodes = emgold["Cathodes"].astype(str).unique()

    fig, axes = plt.subplots(
        ncols=8,
        nrows=7,
        figsize=(17, 10),
        gridspec_kw={"width_ratios": [4, 4, 1, 3, 3, 3, 3, 1]})
    gs = axes[0, 7].get_gridspec()
    for ax in axes[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, -1])
    for ax in axes[0:, 2]:
        ax.remove()
    axbig1 = fig.add_subplot(gs[0:, 2])

    cmap_ = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "#fa525b"])
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "dodgerblue"])
    for i in range(7):
        for j in range(8):
            axes[i][j].set_frame_on(False)
            axes[i][j].set_xlabel(" ")
            axes[i][j].set_ylabel(" ")
            axes[i][j].set_yticks([])
            axes[i][j].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            axes[i][j].set_xticks([])
    for i, muscle in enumerate(muscles):
        data = emg[emg["ElectrodeConftxt"] == conf]
        df2 = pd.crosstab(
            emg["Frequency"],
            emg["Amplitude"],
            values=data["".join([muscle, "max"])],
            aggfunc="mean",
            dropna=False).fillna(0)
        sns.heatmap(
            df2,
            annot=False,
            cmap=cmap_,
            cbar_ax=axbig1,
            ax=axes[i - (i // 7) * 7, i // 7],
            vmin=0,
            vmax=100,
            mask=False,
            cbar_kws={"aspect": 100, "label": "Max muscle activation [%]"})
        axes[i - (i // 7) * 7, i // 7].set_title(muscle, fontsize="small", y=-0.25)
        axes[i - (i // 7) * 7, i // 7].tick_params(axis="both", labelsize="x-small", colors="silver")
        axes[i - (i // 7) * 7, i // 7].tick_params(axis="y", labelrotation=0)
        if not (i // 7 == 0):
            axes[i - (i // 7) * 7, i // 7].set_yticks([])
        if not (i - (i // 7) * 7 == 0):
            axes[i - (i // 7) * 7, i // 7].set_xticks([])
    row = [1, 1, 3, 1, 1, 3]
    col = [3, 4, 3, 6, 5, 6]
    for i, angle in enumerate(angles):
        data = kin[kin["ElectrodeConftxt"] == conf]
        df2 = pd.crosstab(
            kin["Frequency"],
            kin["Amplitude"],
            values=data[angle].abs(),
            aggfunc="mean",
            dropna=False).fillna(0)

        sns.heatmap(
            df2,
            annot=False,
            cmap=cmap2,
            cbar_ax=axbig,
            ax=axes[row[i], col[i]],
            vmin=0,
            vmax=kin[kin.columns[-6:]].max().max(),
            mask=False,
            cbar_kws={"aspect": 100, "label": "Max angle [degree]"})
        axes[row[i], col[i]].set_title(angle[:-3], fontsize="small", y=-0.25)
        axes[row[i], col[i]].tick_params(axis="both", labelsize="x-small", colors="silver")
        axes[row[i], col[i]].tick_params(axis="y", labelrotation=0)
        if not (col[i] == 3):
            axes[row[i], col[i]].set_yticks([])
        if not (row[i] == 0):
            axes[row[i], col[i]].set_xticks([])

    for i in range(7):
        for j in range(8):
            axes[i][j].set_frame_on(False)
            axes[i][j].set_xlabel(" ")
            axes[i][j].set_ylabel(" ")

    fig.suptitle("Amplitude [mA]", color="w", fontsize="large")
    fig.supylabel("Frequency [Hz]", color="w", fontsize="large")
    plt.tight_layout()
    f = (r"c://Users/axell/Desktop/Master_Thesis/little_RNN/src/Datainsight/BestConf/"+ conf.replace(" ", "").replace(",", "").replace(":", "")+ ".png")
    fig.savefig(f)


def draw_per_confless(emgdf, conf, angles, muscles):

    fig, axes = plt.subplots(
        ncols=7,
        nrows=7,
        figsize=(17, 10),
        gridspec_kw={"width_ratios": [3, 4, 4, 1, 4, 4, 1]})
    gs = axes[0, 6].get_gridspec()
    for ax in axes[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, -1])
    for ax in axes[0:, 3]:
        ax.remove()
    axbig1 = fig.add_subplot(gs[0:, 3])
    for ax in axes[0:, 0]:
        ax.remove()
    axelec = fig.add_subplot(gs[0:6, 0])

    background_color = "#323335"  # 'w'
    writings = "w"
    cmap_ = matplotlib.colors.LinearSegmentedColormap.from_list("", [background_color, "#fa525b"])
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", [background_color, "#19D3C5"])
    for i in range(7):
        for j in range(6):
            axes[i][j + 1].set_frame_on(False)
            axes[i][j + 1].set_xlabel(" ")
            axes[i][j + 1].set_ylabel(" ")
            axes[i][j + 1].set_yticks([])
            axes[i][j + 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            axes[i][j + 1].set_xticks([])
    for i, muscle in enumerate(muscles):
        data = emg[emg["ElectrodeConftxt"] == conf]
        df2 = pd.crosstab(
            emg["Frequency"],
            emg["Amplitude"],
            values=data["".join([muscle, "max"])],
            aggfunc="mean",
            dropna=False).fillna(0)
        sns.heatmap(
            df2,
            annot=False,
            cmap=cmap_,
            cbar_ax=axbig1,
            ax=axes[i - (i // 7) * 7, i // 7 + 1],
            vmin=0,
            vmax=100,
            mask=False,
            cbar_kws={"aspect": 100, "label": "Max muscle activation [%]"})
        axes[i - (i // 7) * 7, i // 7 + 1].set_title(muscle, fontsize="small", y=-0.25, color=writings)
        axes[i - (i // 7) * 7, i // 7 + 1].tick_params(axis="both", labelsize="x-small", colors=writings)
        axes[i - (i // 7) * 7, i // 7 + 1].tick_params(axis="y", labelrotation=0)

        if not (i // 7 + 1 == 0 + 1):
            axes[i - (i // 7) * 7, i // 7 + 1].set_yticks([])
        else:
            axes[i - (i // 7) * 7, i // 7 + 1].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5],labels=["20", "40", "60", "80", "100", "120"])
        if not (i - (i // 7) * 7 == 0):
            axes[i - (i // 7) * 7, i // 7 + 1].set_xticks([])

        axbig1.set_ylabel("Max muscle activation [%]", color=writings, fontsize="xx-large")
        axes[3, 1].set_ylabel("Frequency [Hz]", color=writings, fontsize="x-large")

    row = [1, 3, 1, 3]
    col = [3, 3, 4, 4]

    for i, angle in enumerate(angles):
        data = kin[kin["ElectrodeConftxt"] == conf]
        df2 = pd.crosstab(
            kin["Frequency"],
            kin["Amplitude"],
            values=data[angle].abs(),
            aggfunc="mean",
            dropna=False).fillna(0)

        sns.heatmap(
            df2,
            annot=False,
            cmap=cmap2,
            cbar_ax=axbig,
            ax=axes[row[i], col[i] + 1],
            vmin=0,
            vmax=kin[kin.columns[-6:]].max().max(),
            mask=False,
            cbar_kws={"aspect": 100, "label": "Max angle [degree]"})
        axes[row[i], col[i] + 1].set_title(angle[:-3], fontsize="small", y=-0.25, color=writings)
        axes[row[i], col[i] + 1].tick_params(axis="both", labelsize="x-small", colors=writings, color=writings)
        axes[row[i], col[i] + 1].tick_params(axis="y", labelrotation=0)

        axbig.set_ylabel("Max angle [degree]", color=writings, fontsize="xx-large")
        if not (col[i] == 3):
            axes[row[i], col[i] + 1].set_yticks([])
        else:
            axes[row[i], col[i] + 1].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5],labels=["20", "40", "60", "80", "100", "120"])
        if not (row[i] == 0):
            axes[row[i], col[i] + 1].set_xticks([])

    for i in range(7):
        for j in range(6):
            axes[i][j + 1].set_frame_on(False)
            axes[i][j + 1].set_xlabel(" ")
            axes[i][j + 1].set_ylabel(" ")
    axes[0, 1].set_xlabel("LEFT", color=writings, fontsize="xx-large", labelpad=-125)
    axes[0, 1].set_xlabel("LEFT", color=writings, fontsize="xx-large", labelpad=-125)
    axes[0, 2].set_xlabel("RIGHT", color=writings, fontsize="xx-large", labelpad=-125)
    axes[0, 4].set_xlabel("LEFT", color=writings, fontsize="xx-large", labelpad=-125)
    axes[0, 5].set_xlabel("RIGHT", color=writings, fontsize="xx-large", labelpad=-125)
    axes[3, 1].set_ylabel("Frequency [Hz]", color=writings, fontsize="x-large")
    axbig1.tick_params(axis="y", labelsize="large", colors=writings, color="black")
    axbig.tick_params(axis="y", labelsize="large", colors=writings, color="black")

    cathode = data["Cathodes"].astype(str).iloc[0]
    anodes = (data["Anodes"].astype(str).iloc[0].replace("[", "").replace("]", "").split(", "))
    anodes = [int(s) for s in anodes]
    plot_electrode(axelec, int(cathode.replace("[", "").replace("]", "")), anodes)

    fig.suptitle("Amplitude [mA]", color=writings, fontsize="x-large", y=0.93, x=0.28)
    fig.text(0.68, 0.91, "Amplitude [mA]", color=writings, fontsize="x-large")
    plt.tight_layout()
    f = (r"c://Users/axell/Desktop/Master_Thesis/little_RNN/srcnew/Datainsight/BestConf2/"+ conf.replace(" ", "").replace(",", "").replace(":", "")+ ".svg")
    fig.savefig(f, transparent=True)


def plot_electrode(ax: matplotlib.axes.Axes, cathode: int, anodes: List[int]):
    with open(os.path.abspath("../images/electrode.png"), "rb") as electrode_file:
        electrode_im = plt.imread(electrode_file)
    height, width = electrode_im.shape[0], electrode_im.shape[1]
    x_offset = 4
    y_offset = 90
    x0 = ELECTRODE_POSITIONS_ELEC[cathode][0] * width + x_offset
    y0 = ELECTRODE_POSITIONS_ELEC[cathode][1] * height + y_offset

    # Cathode
    image = plt.imread(os.path.abspath("../images/cathode.png"))
    image_box = OffsetImage(image, zoom=0.15)
    ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
    rect3 = matplotlib.patches.Rectangle(
        (x0 - 14, y0 - 40), 36, 80, clip_box=ab, color="#FA525B", capstyle="round"
    )
    ax.add_patch(rect3)
    pos = (x0 - 14 + int(36 / 2), y0 - 40 + int(80 / 2))
    ax.text(
        pos[0],
        pos[1],
        "-",
        fontsize="xx-large",
        color="snow",
        horizontalalignment="center",
        verticalalignment="center",
    )

    if not (anodes == None):
        for an in anodes:
            x0 = ELECTRODE_POSITIONS_ELEC[an][0] * width + x_offset
            y0 = ELECTRODE_POSITIONS_ELEC[an][1] * height + y_offset

            # Cathode
            image = plt.imread(os.path.abspath("../images/anode.png"))
            image_box = OffsetImage(image, zoom=0.15)
            ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
            rect3 = matplotlib.patches.Rectangle(
                (x0 - 14, y0 - 40),
                36,
                80,
                clip_box=ab,
                color="#8A8A8C",
                capstyle="round",
            )
            ax.add_patch(rect3)
            pos = (x0 - 14 + int(36 / 2), y0 - 40 + int(80 / 2))
            ax.text(
                pos[0],
                pos[1],
                "+",
                fontsize="xx-large",
                color="snow",
                horizontalalignment="center",
                verticalalignment="center",
            )

    ax.imshow(electrode_im, aspect="auto")
    ax.set_frame_on(False)
    ax.set_xlim(60, 260)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def draw_per_cathodeALL(emgdf, cathode, pulses, angles, muscles):

    fig, axes = plt.subplots(
        ncols=9,
        nrows=7,
        figsize=(17, 10),
        gridspec_kw={"width_ratios": [5, 5, 5, 1, 3, 3, 3, 3, 1]})
    gs = axes[0, 8].get_gridspec()
    for ax in axes[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, -1])
    for ax in axes[0:, 3]:
        ax.remove()
    axbig1 = fig.add_subplot(gs[0:, 3])
    for ax in axes[0:, 0]:
        ax.remove()
    axelec = fig.add_subplot(gs[0:6, 0])
    cmap_ = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "#fa525b"])
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "dodgerblue"])
    for i in range(7):
        for j in range(8):
            axes[i][j + 1].set_frame_on(False)
            axes[i][j + 1].set_xlabel(" ")
            axes[i][j + 1].set_ylabel(" ")
            axes[i][j + 1].set_yticks([])
            axes[i][j + 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            axes[i][j + 1].set_xticks([])
    for i, muscle in enumerate(muscles):
        data1 = emgdf[emgdf.Pulses == pulses]
        data = data1[data1["Cathodes"].astype(str) == cathode]
        df2 = pd.crosstab(
            emg["Frequency"],
            emg["Amplitude"],
            values=data["".join([muscle, "max"])],
            aggfunc="mean",
            dropna=False).fillna(0)
        sns.heatmap(
            df2,
            annot=False,
            cmap=cmap_,
            cbar_ax=axbig1,
            ax=axes[i - (i // 7) * 7, i // 7 + 1],
            vmin=0,
            vmax=100,
            mask=False,
            cbar_kws={"aspect": 100, "label": "Max muscle activation [%]"})
        axes[i - (i // 7) * 7, i // 7 + 1].set_title(muscle, fontsize="small", y=-0.25)
        axes[i - (i // 7) * 7, i // 7 + 1].tick_params(axis="both", labelsize="x-small", colors="silver")
        axes[i - (i // 7) * 7, i // 7 + 1].tick_params(axis="y", labelrotation=0)
        if not (i // 7 + 1 == 0 + 1):
            axes[i - (i // 7) * 7, i // 7 + 1].set_yticks([])
        if not (i - (i // 7) * 7 == 0):
            axes[i - (i // 7) * 7, i // 7 + 1].set_xticks([])
    row = [1, 1, 3, 1, 1, 3]
    col = [3, 4, 3, 6, 5, 6]
    for i, angle in enumerate(angles):
        data1 = kin[kin.Pulses == pulses]
        data = data1[data1["Cathodes"].astype(str) == cathode]
        df2 = pd.crosstab(
            kin["Frequency"],
            kin["Amplitude"],
            values=data[angle].abs(),
            aggfunc="mean",
            dropna=False).fillna(0)

        sns.heatmap(
            df2,
            annot=False,
            cmap=cmap2,
            cbar_ax=axbig,
            ax=axes[row[i], col[i] + 1],
            vmin=0,
            vmax=kin[kin.columns[-6:]].max().max(),
            mask=False,
            cbar_kws={"aspect": 100, "label": "Max angle [degree]"})
        axes[row[i], col[i] + 1].set_title(angle[:-3], fontsize="small", y=-0.25)
        axes[row[i], col[i] + 1].tick_params(axis="both", labelsize="x-small", colors="silver")
        axes[row[i], col[i] + 1].tick_params(axis="y", labelrotation=0)
        if not (col[i] == 3):
            axes[row[i], col[i] + 1].set_yticks([])
        if not (row[i] == 0):
            axes[row[i], col[i] + 1].set_xticks([])

    for i in range(7):
        for j in range(8):
            axes[i][j + 1].set_frame_on(False)
            axes[i][j + 1].set_xlabel(" ")
            axes[i][j + 1].set_ylabel(" ")
    plot_electrode(axelec, int(cathode.replace("[", "").replace("]", "")), None)

    fig.suptitle("Amplitude [mA]", color="w", fontsize="large")
    fig.supylabel("Frequency [Hz]", color="w", fontsize="large")
    plt.tight_layout()
    f = (r"c://Users/axell/Desktop/Master_Thesis/little_RNN/src/Datainsight/BestCathPulse/Cathode"+ cathode.replace("[", "").replace("]", "")+ "pulse"+ str(pulses)+ ".png")
    fig.savefig(f)


def draw_per_cathodeALLless(emgdf, cathode, pulses, angles, muscles):

    fig, axes = plt.subplots(
        ncols=7,
        nrows=7,
        figsize=(17, 10),
        gridspec_kw={"width_ratios": [3, 4, 4, 1, 4, 4, 1]})
    gs = axes[0, 6].get_gridspec()
    for ax in axes[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, -1])
    for ax in axes[0:, 3]:
        ax.remove()
    axbig1 = fig.add_subplot(gs[0:, 3])
    for ax in axes[0:, 0]:
        ax.remove()
    axelec = fig.add_subplot(gs[0:6, 0])

    background_color = "#323335"
    cmap_ = matplotlib.colors.LinearSegmentedColormap.from_list("", [background_color, "#fa525b"])
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", [background_color, "dodgerblue"])
    for i in range(7):
        for j in range(6):
            axes[i][j + 1].set_frame_on(False)
            axes[i][j + 1].set_xlabel(" ")
            axes[i][j + 1].set_ylabel(" ")
            axes[i][j + 1].set_yticks([])
            axes[i][j + 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            axes[i][j + 1].set_xticks([])
    for i, muscle in enumerate(muscles):
        data1 = emgdf[emgdf.Pulses == pulses]
        data = data1[data1["Cathodes"].astype(str) == cathode]
        df2 = pd.crosstab(
            emg["Frequency"],
            emg["Amplitude"],
            values=data["".join([muscle, "max"])],
            aggfunc="mean",
            dropna=False).fillna(0)
        sns.heatmap(
            df2,
            annot=False,
            cmap=cmap_,
            cbar_ax=axbig1,
            ax=axes[i - (i // 7) * 7, i // 7 + 1],
            vmin=0,
            vmax=100,
            mask=False,
            cbar_kws={"aspect": 100, "label": "Max muscle activation [%]"})
        axes[i - (i // 7) * 7, i // 7 + 1].set_title(muscle, fontsize="small", y=-0.25)
        axes[i - (i // 7) * 7, i // 7 + 1].tick_params(axis="both", labelsize="x-small", colors="white")
        axes[i - (i // 7) * 7, i // 7 + 1].tick_params(axis="y", labelrotation=0)

        if not (i // 7 + 1 == 0 + 1):
            axes[i - (i // 7) * 7, i // 7 + 1].set_yticks([])
        else:
            axes[i - (i // 7) * 7, i // 7 + 1].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5],labels=["20", "40", "60", "80", "100", "120"])
        if not (i - (i // 7) * 7 == 0):
            axes[i - (i // 7) * 7, i // 7 + 1].set_xticks([])
    row = [1, 3, 1, 3]
    col = [3, 3, 4, 4]

    for i, angle in enumerate(angles):
        data1 = kin[kin.Pulses == pulses]
        data = data1[data1["Cathodes"].astype(str) == cathode]
        df2 = pd.crosstab(
            kin["Frequency"],
            kin["Amplitude"],
            values=data[angle].abs(),
            aggfunc="mean",
            dropna=False).fillna(0)

        sns.heatmap(
            df2,
            annot=False,
            cmap=cmap2,
            cbar_ax=axbig,
            ax=axes[row[i], col[i] + 1],
            vmin=0,
            vmax=kin[kin.columns[-6:]].max().max(),
            mask=False,
            cbar_kws={"aspect": 100, "label": "Max angle [degree]"})
        axes[row[i], col[i] + 1].set_title(angle[:-3], fontsize="small", y=-0.25)
        axes[row[i], col[i] + 1].tick_params(axis="both", labelsize="x-small", colors="white")
        axes[row[i], col[i] + 1].tick_params(axis="y", labelrotation=0)
        if not (col[i] == 3):
            axes[row[i], col[i] + 1].set_yticks([])
        else:
            axes[row[i], col[i] + 1].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5],labels=["20", "40", "60", "80", "100", "120"])
        if not (row[i] == 0):
            axes[row[i], col[i] + 1].set_xticks([])

    for i in range(7):
        for j in range(6):
            axes[i][j + 1].set_frame_on(False)
            axes[i][j + 1].set_xlabel(" ")
            axes[i][j + 1].set_ylabel(" ")
    plot_electrode(axelec, int(cathode.replace("[", "").replace("]", "")), None)
    fig.suptitle("Amplitude [mA]", color="w", fontsize="large")
    plt.ylabel("Frequency [Hz]", color="w", fontsize="large", labelpad=-300)
    plt.tight_layout()
    f = (r"c://Users/axell/Desktop/Master_Thesis/little_RNN/src/Datainsight/BestCathPulse2/Cathode"+ cathode.replace("[", "").replace("]", "")+ "pulse"+ str(pulses)+ ".png")
    fig.savefig(f, transparent=True)


def draw_per_cathodeALL3(emgdf, cathode, pulses, angles, muscles):

    fig, axes = plt.subplots(ncols=4, nrows=7, figsize=(8, 12), gridspec_kw={"width_ratios": [3, 4, 4, 1]})
    gs = axes[0, 3].get_gridspec()
    for ax in axes[0:, 3]:
        ax.remove()
    axbig1 = fig.add_subplot(gs[0:, 3])
    for ax in axes[0:, 0]:
        ax.remove()
    axelec = fig.add_subplot(gs[1:5, 0])

    background_color = "w"
    cmap_ = matplotlib.colors.LinearSegmentedColormap.from_list("", [background_color, "#fa525b"])
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", [background_color, "#19D3C5"])
    for i in range(7):
        for j in range(3):
            axes[i][j + 1].set_frame_on(False)
            axes[i][j + 1].set_xlabel(" ")
            axes[i][j + 1].set_ylabel(" ")
            axes[i][j + 1].set_yticks([])
            axes[i][j + 1].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            axes[i][j + 1].set_xticks([])
    for i, muscle in enumerate(muscles):
        data1 = emgdf[emgdf.Pulses == pulses]
        data = data1[data1["Cathodes"].astype(str) == cathode]
        df2 = pd.crosstab(
            emg["Frequency"],
            emg["Amplitude"],
            values=data["".join([muscle, "max"])],
            aggfunc="mean",
            dropna=False).fillna(0)
        sns.heatmap(
            df2,
            annot=False,
            cmap=cmap_,
            cbar_ax=axbig1,
            ax=axes[i - (i // 7) * 7, i // 7 + 1],
            vmin=0,
            vmax=100,
            mask=False,
            cbar_kws={"aspect": 100, "label": "Max muscle activation [%]"})
        axes[i - (i // 7) * 7, i // 7 + 1].set_title(muscle, fontsize="small", y=-0.25, color="black")
        axes[i - (i // 7) * 7, i // 7 + 1].tick_params(axis="both", labelsize="small", colors="black", color="black")
        axes[i - (i // 7) * 7, i // 7 + 1].tick_params(axis="y", labelrotation=0, colors="black", color="black")
        axbig1.tick_params(axis="y", labelsize="large", colors="black", color="black")
        axbig1.set_ylabel("Max muscle activation [%]", color="black", fontsize="xx-large")
        axes[0, 1].set_xlabel("LEFT", color="black", fontsize="xx-large", labelpad=-165)
        axes[0, 2].set_xlabel("RIGHT", color="black", fontsize="xx-large", labelpad=-165)
        axes[3, 1].set_ylabel("Frequency [Hz]", color="black", fontsize="x-large")

        if not (i // 7 + 1 == 0 + 1):
            axes[i - (i // 7) * 7, i // 7 + 1].set_yticks([])
        else:
            axes[i - (i // 7) * 7, i // 7 + 1].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5],labels=["20", "40", "60", "80", "100", "120"])
        if not (i - (i // 7) * 7 == 0):
            axes[i - (i // 7) * 7, i // 7 + 1].set_xticks([])
    row = [1, 3, 1, 3]
    col = [3, 3, 4, 4]

    for i, angle in enumerate(angles):
        data1 = kin[kin.Pulses == pulses]
        data = data1[data1["Cathodes"].astype(str) == cathode]

    plot_electrode(axelec, int(cathode.replace("[", "").replace("]", "")), None)

    fig.suptitle("Amplitude [mA]", color="black", fontsize="x-large", y=0.92)
    plt.ylabel("Frequency [Hz]", color="black", fontsize="large", labelpad=30)
    plt.tight_layout()

    f = (r"c://Users/axell/Desktop/Master_Thesis/little_RNN/srcnew/Datainsight/BestCathPulse3/Cathode"+ cathode.replace("[", "").replace("]", "")+ "pulse"+ str(pulses)+ ".svg")
    fig.savefig(f, transparent=True)


def draw_per_1confelse(emgdf, kin, conf, angles, muscles):

    freqs = list(emgold["Frequency"].unique())
    amps = list(emgold["Amplitude"].unique())

    fig, axes = plt.subplots(
        ncols=2 * len(freqs) + 2,
        nrows=len(amps),
        figsize=(17, 10),
        gridspec_kw={"width_ratios": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1]})
    gs = axes[0, 2 * len(freqs) + 1].get_gridspec()
    for ax in axes[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, -1])  # for kin
    for ax in axes[0:, 2]:
        ax.remove()
    axbig1 = fig.add_subplot(gs[0:, -2])  # for muscles
    cmap_ = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "#fa525b"])
    cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "dodgerblue"])
    for i in range(len(amps)):
        for j in range(2 * len(freqs) + 2):
            axes[i][j].set_frame_on(False)
            axes[i][j].set_xlabel(" ")
            axes[i][j].set_ylabel(" ")
            axes[i][j].set_yticks([])
            axes[i][j].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            axes[i][j].set_xticks([])

    data = emgdf[emgdf["ElectrodeConftxt"] == conf]
    datakin = kin[kin["ElectrodeConftxt"] == conf]
    i = -1
    j = -1

    for freq in freqs:
        j = j + 1
        gooddata1 = data[data.Frequency == freq]
        gooddata1kin = datakin[datakin.Frequency == freq]
        for amp in amps:
            i = i + 1
            df2 = pd.DataFrame(
                {
                    "left": [0, 0, 0, 0, 0, 0, 0],
                    "right": [0, 0, 0, 0, 0, 0, 0],
                    "leftang": [0, 0, 0, 0, 0, 0, 0],
                    "leftAdd": [0, 0, 0, 0, 0, 0, 0],
                    "rightAdd": [0, 0, 0, 0, 0, 0, 0],
                    "rightang": [0, 0, 0, 0, 0, 0, 0],
                }
            )
            gooddata = gooddata1[gooddata1.Amplitude == amp]
            gooddatakin = gooddata1kin[gooddata1kin.Amplitude == amp]

            if len(gooddata) != 0:
                df2["left"] = gooddata[gooddata.columns[-14:-7]].values[0]
                df2["right"] = gooddata[gooddata.columns[-7:]].values[0]
                for i in [0, 1]:

                    df2["leftang"].iloc[i] = gooddatakin.LHIPFmax.values.mean()
                    df2["rightang"].iloc[i] = gooddatakin.RHIPFmax.values.mean()
                    df2["leftAdd"].iloc[i] = gooddatakin.LAddmax.values.mean()
                    df2["rightAdd"].iloc[i] = gooddatakin.RAddmax.values.mean()
                    df2["leftang"].iloc[i + 3] = gooddatakin.LKNEXmax.values.mean()
                    df2["rightang"].iloc[i + 3] = gooddatakin.RKNEXmax.values.mean()

                sns.heatmap(
                    df2[["left", "right"]],
                    annot=False,
                    cmap=cmap_,
                    cbar_ax=axbig1,
                    ax=axes[i][2 * j],
                    vmin=0,
                    vmax=100,
                    mask=False,
                    cbar_kws={"aspect": 100, "label": "Max angle [degree]"})
                sns.heatmap(
                    df2[["leftang", "leftAdd", "rightAdd", "rightang"]],
                    annot=False,
                    cmap=cmap2,
                    cbar_ax=axbig,
                    ax=axes[i][2 * j + 1],
                    vmin=0,
                    vmax=kin[kin.columns[-6:]].max().max(),
                    mask=False,
                    cbar_kws={"aspect": 100, "label": "Max angle [degree]"})

    for i in range(len(amps)):
        for j in range(2 * len(freqs)):
            if not (j == 0):
                axes[i][j].set_yticks([])
            if not (i == 0):
                axes[i][j].set_xticks([])
            axes[i][j].set_frame_on(False)
            axes[i][j].set_xlabel(" ")
            axes[i][j].set_ylabel(" ")

    fig.suptitle("Amplitude [mA]", color="w", fontsize="large")
    fig.supylabel("Frequency [Hz]", color="w", fontsize="large")
    plt.tight_layout()
    f = (r"c://Users/axell/Desktop/Master_Thesis/little_RNN/src/Datainsight/BestConfFA/"+ conf.replace(" ", "").replace(",", "").replace(":", "")+ ".png")
    fig.savefig(f)

##### UNCOMMENT THE LINE YOU WANT TO USE : MUSCLES

MUSCLES = ["LIl","LRF","LVLat","LST","LTA","LMG","LSol","RIl","RRF","RVLat","RST","RTA","RMG","RSol"]
# for musc in MUSCLES:
#    draw_per_Cath(emg, musc)
#    draw_allconf(emg, musc)
#    draw_per_CathSPAT(emg, musc)

##### UNCOMMENT THE LINE YOU WANT TO USE : ANGLES

# ANGLES = ['LHIPFmax', 'LAddmax', 'LKNEXmax', 'RHIPFmax', 'RAddmax', 'RKNEXmax']
ANGLES = ["LHIPFmax", "LKNEXmax", "RHIPFmax", "RKNEXmax"]
# for angle in ANGLES :
#    draw_per_CathAngle(kin, angle)

CONF = list(emgold["ElectrodeConftxt"].unique())

##### UNCOMMENT THE LINE YOU WANT TO USE : CONFIGURATION

for config in CONF:
    # draw_per_1conf(emg,list(emgold['ElectrodeConftxt'].unique())[0], ANGLES, MUSCLES)
    # draw_per_1conf(emg,config, ANGLES, MUSCLES)
    # draw_per_1confelse(emg,kin, list(emgold['ElectrodeConftxt'].unique())[0], ANGLES, MUSCLES)
    # draw_per_1confelse(emg,kin, config, ANGLES, MUSCLES)
    draw_per_confless(emg, config, ANGLES, MUSCLES)

##### UNCOMMENT THE LINE YOU WANT TO USE : CATHODES

cathodes = emg["Cathodes"].astype(str).unique()
# draw_per_cathodeALL(emg, cathodes[0], 1, ANGLES, MUSCLES)
# for cathode in cathodes:
# draw_per_cathodeALL3(emg, cathode, 1, ANGLES, MUSCLES)

# draw_per_cathodeALL(emg, cathode, 1, ANGLES, MUSCLES)
# draw_per_cathodeALL(emg, cathode, 3, ANGLES, MUSCLES)
# cathodes = emgdf['Cathodes'].astype(str).unique()


###   Plots  ###
fig = plt.figure()
emg_trial = list(emg[emg.columns[-28:-14]].iloc[0, :].values)

timeemg = np.linspace(0, emg_trial[0].shape[0] - 1, emg_trial[0].shape[0])
for side in [1]:
    for r in range(7):
        plt.plot(
            timeemg,
            2 * emg_trial[r + 7 * (side)] - r * 230,
            linewidth=1,
            color="lightgrey",
        )
f = "c://Users/axell/Desktop/Master_Thesis/little_RNN/src/checkup.png"
fig.savefig(f)
