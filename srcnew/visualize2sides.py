# Code for .gif of the two legs of the patient
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
import numpy as np
import pandas as pd
import matplotlib
from utils import plot_electrode_activation
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import List, Tuple
from data import PATH, N_ELECTRODES, MUSCLES
from utils import ELECTRODE_POSITIONS_ELEC
from matplotlib.patches import Rectangle
import scipy.io

###################################################

parser = argparse.ArgumentParser(description="Train an RNN on subject data")
parser.add_argument("ID", metavar="N", type=str, help="ID of the test")
args = vars(parser.parse_args())

ID = args["ID"]
# loading
kin1 = cPickle.load(open("kin4.pkl", "rb"))

ID = "lstmnewdata123"
selected_trial = 13
samplingf = 1481.48
train_sets, train_targets = torch.load(f"{data.PATH}/{ID}/train_sets_targets.pt")
testing_sets1, testing_targets1 = torch.load(f"{data.PATH}/{ID}/test_sets_targets.pt")
test_stim_features1 = torch.load(f"{data.PATH}/{ID}/test_stim_features.pt")
pred1 = torch.load(f"{data.PATH}/{ID}/test_pred.pt")

# Rematch everything
test_stim_features = pd.DataFrame(columns=test_stim_features1.columns)
targ = []
pred = []
sets = []
kin = kin1.copy()
for i in range(kin1.shape[0]):
    rowfeat = test_stim_features1[(test_stim_features1[["Frequency", "Amplitude", "PulseWidth", "Pulses", "Cathodes", "Anodes"]] == pd.Series(list(kin1[["Frequency","Amplitude","PulseWidth","Pulses","Cathodes","Anodes"]].iloc[i].values),index=["Frequency","Amplitude","PulseWidth","Pulses","Cathodes","Anodes"],)).all(axis="columns")][test_stim_features.columns]
    if not (rowfeat.empty):
        test_stim_features = test_stim_features.append(
            rowfeat.iloc[0], ignore_index=False
        )
        targ.append(testing_targets1[rowfeat.index[0], :, :])
        pred.append(pred1[rowfeat.index[0], :, :])
        sets.append(testing_sets1[rowfeat.index[0], :, :, :])
    else:
        kin = kin.drop(index=i)
testing_sets = np.array(list(sets))
testing_targets = np.array(list(targ))
preds = np.array(list(pred))
###################################################
# Reconstruct kinematics dataframe
markers = ["LTOE", "LANK", "LKNE", "LHIP", "RHIP", "RKNE", "RANK", "RTOE"]
cols = ["Trial", "Frame"]
cols.extend(markers)
df_kinematics = pd.DataFrame(columns=cols)
df_kinematics.index = df_kinematics["Trial"]

# Setting up Data Set for Animation
numDataPoints = np.array(list(kin.iloc[0, 11:19].values)).shape[1]
emg = testing_targets

# Resample for matching sizes
deletable = []
for i in range(kin.shape[0]):
    if kin.RTOE.iloc[i].shape[0] < 3:
        deletable.append(i)
kin = kin.drop(kin.index[deletable])
test_stim_features = test_stim_features.drop(test_stim_features.index[deletable])
emg = np.delete(emg, deletable, 0)
preds = np.delete(preds, deletable, 0)

df_kinematics = kin
emg_trial = emg[selected_trial].T
pred_trial = preds[selected_trial].T

emg_trial = resample(
    emg_trial.T, n_samples=kin.RTOE.iloc[selected_trial].shape[0], random_state=0
)
pred_trial = resample(
    pred_trial.T, n_samples=kin.RTOE.iloc[selected_trial].shape[0], random_state=0
)
###################################################
# Functions for plotting


def animate_func(num):
    ax[3].clear()
    ax[1].clear()
    ax[2].clear()
    # Patient's body
    ax[3].plot_surface(x_, y_, z_, color="#9D9B9B", antialiased=True, shade=True)
    # Patient's head
    ax[3].plot_surface(x, y, z, color="#9D9B9B", antialiased=True, shade=True)
    ax[3].plot3D(
        marker_data[0:4, num, 0],
        marker_data[0:4, num, 1],
        marker_data[0:4, num, 2],
        color="#fa525b",
        antialiased=True,
        linewidth=12,
        fillstyle="full",
    )
    ax[3].plot3D(
        marker_data[4:8, num, 0],
        marker_data[4:8, num, 1],
        marker_data[4:8, num, 2],
        color="#fa525b",
        antialiased=True,
        linewidth=12,
        fillstyle="full",
    )

    # Setting Axes Limits
    ax[3].set_xlim3d([-700, 700])
    ax[3].set_ylim3d([-1100, 200])
    ax[3].set_zlim3d([0, 1300])

    # Adding Figure Labels
    ax[3].set_title(
        "Trial "
        + str(selected_trial)
        + "\nFrame = "
        + str(int(df_kinematics.Frame.iloc[selected_trial][num]))
    )
    ax[3].set_xlabel("x")
    ax[3].set_ylabel("y")
    ax[3].set_zlabel("z")
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_zticks([])
    # Make the panes transparent
    ax[3].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[3].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # Make the grid lines transparent
    ax[3].xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax[3].yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax[3].zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

    #######################################################
    # EMG PLOTS
    ax[1].set_frame_on(False)
    ax[1].set_title("LEFT", fontsize="15")
    ax[2].set_frame_on(False)
    ax[2].set_title("RIGHT", fontsize="15")

    start = int(num - window_width / 2)
    end = int(num + window_width / 2)
    # Left
    ax[1].plot(
        time[start + shift : end + shift],
        2 * exp_stim[start + shift : end + shift, :] * 0.03 + 2,
        color="#fa525b",
    )

    for r in range(7):
        true_line = ax[1].plot(
            time[start + shift : end + shift],
            trial[start + shift : end + shift, r] * 0.04 - r * 2,
            color="#dbdbdd",
            linewidth=1,
        )
        predicted_line = ax[1].plot(
            time[start + shift : end + shift],
            pred_trial[start + shift : end + shift, r] * 0.04 - r * 2,
            "--",
            color="#dbdbdd",
            alpha=0.8,
            linewidth=1.8,
        )
        ax[1].plot([0, 0], [-13, 1], linestyle="dotted", color="#fa525b", linewidth=0.5)
        ax[1].set_frame_on(False)
        ax[1].set_xticks(ticks, lab)
        ax[1].set_ylim(-13, 3)
        ax[1].set_xlim(time[start + shift], time[end + shift])
        ax[1].tick_params("x", labelbottom=False)

    ax[1].tick_params("x", labelbottom=True, labelsize="10")
    ax[1].set_yticklabels((" ", "Sol", "TA", "MG", "ST", "VLat", "RF", "Add", " "))
    ax[1].set_xlabel("Time [ms]", fontsize="10")
    #####
    # Right
    ax[2].plot(
        time[start + shift : end + shift],
        2 * exp_stim[start + shift : end + shift, :] * 0.03 + 2,
        color="#fa525b",
    )
    for r in range(7):
        true_line = ax[2].plot(
            time[start + shift : end + shift],
            trial[start + shift : end + shift, r] * 0.04 - r * 2,
            color="#dbdbdd",
            linewidth=1,
        )
        predicted_line = ax[2].plot(
            time[start + shift : end + shift],
            pred_trial[start + shift : end + shift, r] * 0.04 - r * 2,
            "--",
            color="#dbdbdd",
            alpha=0.8,
            linewidth=1.8,
        )
        ax[2].plot([0, 0], [-13, 1], linestyle="dotted", color="#fa525b", linewidth=0.5)
        ax[2].set_frame_on(False)
        ax[2].set_xticks(ticks, lab)
        ax[2].set_ylim(-13, 3)
        ax[2].set_xlim(time[start + shift], time[end + shift])
        ax[2].tick_params("x", labelbottom=False)

    ax[2].tick_params("x", labelbottom=True, labelsize="10")
    ax[2].set_yticks([])
    ax[2].set_xlabel("Time [ms]", fontsize="10")


def plot_electrode(ax: matplotlib.axes.Axes, cathodes: int, anodes: List[int]):
    wid = 36
    hei = 80
    # Anode
    image = plt.imread(os.path.abspath("../images/anode.png"))
    image_box = OffsetImage(image, zoom=0.15)
    for x0, y0 in zip(x_anodes, y_anodes):
        ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
        rect3 = matplotlib.patches.Rectangle(
            (x0 - 14, y0 - 40),
            wid,
            hei,
            clip_box=ab,
            color="#8A8A8C",
            joinstyle="round",
        )
        pos = (x0 - 14 + int(wid / 2), y0 - 40 + int(hei / 2))
        ax.text(
            pos[0],
            pos[1],
            "+",
            fontsize="xx-large",
            color="snow",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.add_patch(rect3)
    # Cathode
    image = plt.imread(os.path.abspath("../images/cathode.png"))
    image_box = OffsetImage(image, zoom=0.15)
    for x0, y0 in zip(x_cathodes, y_cathodes):
        ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
        rect3 = matplotlib.patches.Rectangle(
            (x0 - 14, y0 - 40), 36, 80, clip_box=ab, color="#FA525B", joinstyle="round"
        )
        pos = (x0 - 14 + int(wid / 2), y0 - 40 + int(hei / 2))
        ax.text(
            pos[0],
            pos[1],
            "-",
            fontsize="xx-large",
            color="snow",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.add_patch(rect3)

    ax.imshow(electrode_im)
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


###################################################
# SETTINGS

# For electrode

with open(os.path.abspath("../images/electrode.png"), "rb") as electrode_file:
    electrode_im = plt.imread(electrode_file)
height, width = electrode_im.shape[0], electrode_im.shape[1]
x_offset = 4
y_offset = 90
x_anodes, y_anodes = [], []
for anode in df_kinematics["Anodes"].iloc[selected_trial]:
    x_anodes.append(ELECTRODE_POSITIONS_ELEC[anode][0] * width + x_offset)
    y_anodes.append(ELECTRODE_POSITIONS_ELEC[anode][1] * height + y_offset)
x_cathodes, y_cathodes = [], []
for cathode in df_kinematics["Cathodes"].iloc[selected_trial]:
    x_cathodes.append(ELECTRODE_POSITIONS_ELEC[cathode][0] * width + x_offset)
    y_cathodes.append(ELECTRODE_POSITIONS_ELEC[cathode][1] * height + y_offset)
x_anodes, y_anodes = np.atleast_1d(x_anodes, y_anodes)
x_cathodes, y_cathodes = np.atleast_1d(x_cathodes, y_cathodes)

# For signal

window_width = 30
shift = int(window_width / 2)
ticks = np.linspace(
    -shift,
    emg_trial.shape[0] - 1 + shift,
    int((emg_trial.shape[0] + window_width) / 5) + 1,
)
lab = ["", "", ""]
for i in ticks[3:]:
    lab = np.append(lab, str(int(i)))
time = np.linspace(
    -shift, emg_trial.shape[0] - 1 + shift, emg_trial.shape[0] + window_width
)
arr = np.empty((int(window_width / 2), 13))
arr[:] = 0
stim_duration = emg_trial.shape[0]
stim_arrays = data.create(test_stim_features, stim_duration, fs=samplingf)
stim = stim_arrays[selected_trial, 0, :, :]
arr1 = np.empty((int(window_width / 2), 17))
exp_stim = np.concatenate((arr1, stim, arr1), axis=0)
trial = np.concatenate(
    (arr, emg_trial, arr), axis=0
)  # add zeros before and after emg_trial and time
pred_trial = np.concatenate((arr, pred_trial, arr), axis=0)

# Plotting the Animation
marker_data = np.array(list(df_kinematics.iloc[selected_trial, 11:19].values))
y_ = np.array(
    [
        [
            marker_data[3][0, 1],
            marker_data[3][0, 1],
            -0.1 * marker_data[3][0, 1],
            -0.7 * marker_data[3][0, 1],
        ],
        [
            marker_data[3][0, 1],
            marker_data[3][0, 1],
            -0.1 * marker_data[3][0, 1],
            -0.7 * marker_data[3][0, 1],
        ],
        [
            marker_data[4][0, 1],
            marker_data[4][0, 1],
            -0.3 * marker_data[4][0, 1],
            -0.7 * marker_data[4][0, 1],
        ],
        [
            marker_data[4][0, 1],
            marker_data[4][0, 1],
            -0.3 * marker_data[4][0, 1],
            -0.7 * marker_data[4][0, 1],
        ],
    ]
)
x_ = np.array(
    [
        [
            marker_data[3][0, 0],
            marker_data[3][0, 0],
            1.4 * marker_data[3][0, 0],
            1.5 * marker_data[3][0, 0],
        ],
        [
            marker_data[3][0, 0],
            marker_data[3][0, 0],
            1.3 * marker_data[3][0, 0],
            1.5 * marker_data[3][0, 0],
        ],
        [
            marker_data[4][0, 0],
            marker_data[4][0, 0],
            1.3 * marker_data[4][0, 0],
            1.5 * marker_data[4][0, 0],
        ],
        [
            marker_data[4][0, 0],
            marker_data[4][0, 0],
            1.4 * marker_data[4][0, 0],
            1.5 * marker_data[4][0, 0],
        ],
    ]
)
z_ = np.array(
    [
        [
            0.9 * marker_data[3][0, 2],
            0.9 * marker_data[3][0, 2],
            0.9 * marker_data[3][0, 2],
            0.9 * marker_data[3][0, 2],
        ],
        [
            0.9 * marker_data[3][0, 2],
            1.1 * marker_data[3][0, 2],
            1.1 * marker_data[3][0, 2],
            0.9 * marker_data[3][0, 2],
        ],
        [
            0.9 * marker_data[4][0, 2],
            1.1 * marker_data[4][0, 2],
            1 * marker_data[4][0, 2],
            0.9 * marker_data[4][0, 2],
        ],
        [
            0.9 * marker_data[4][0, 2],
            0.9 * marker_data[4][0, 2],
            0.9 * marker_data[4][0, 2],
            0.9 * marker_data[4][0, 2],
        ],
    ]
)

# Head
r = 185
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:100j, 0.0 : 2.0 * pi : 100j]
x = 0.9 * r * sin(phi) * cos(theta) + (marker_data[3][0, 0] + marker_data[4][0, 0]) / 2
y = 1.1 * r * sin(phi) * sin(theta) - marker_data[4][0, 1]
z = 1.1 * r * cos(phi) + 1 * marker_data[3][0, 2]

###################################################
# Actually plotting
fig, ax = plt.subplots(
    1, 4, figsize=(12, 9), gridspec_kw={"width_ratios": [2, 4, 4, 5]}
)
ax[3] = plt.subplot(144, projection="3d", computed_zorder=False)
ax[2] = plt.subplot(143)
ax[1] = plt.subplot(142)
ax[0] = plt.subplot(141)
plot_electrode(
    ax[0],
    df_kinematics["Cathodes"].iloc[selected_trial],
    df_kinematics["Anodes"].iloc[selected_trial],
)
line_ani = animation.FuncAnimation(
    fig, animate_func, interval=100, frames=numDataPoints
)
a = ax[1].get_xticks
plt.tight_layout()
plt.show()
###################################################

###################################################
# Saving the Animation

f = r"c://Users/yes/Desktop/animate_func.gif"
writergif = animation.PillowWriter(fps=numDataPoints / 10)
line_ani.save(f, writer=writergif)
###################################################
