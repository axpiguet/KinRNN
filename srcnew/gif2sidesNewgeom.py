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
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

###################################################

# Loading the data (here, emg and stim for trial 466)
with open("emgs466.npy", "rb") as f:
    emg = np.load(f, allow_pickle=True)
with open("stimlegs466.npy", "rb") as f:
    stim = np.load(f, allow_pickle=True)

fsemg = 1259.25925925926
fskin = 148.148
stimfs = 10 * fskin
# Patient
l_thigh = 22.0
l_shank = 26.0
# Stim
an = [16]
cath = [2]

##################################################


def to_coo(HIPF, ADD, KNEX):
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
        ankle_x = l_hiptoankle * np.cos(np.radians(alpha)) * np.cos(np.radians(ADD[t]))
        ankle_y = l_hiptoankle * np.cos(np.radians(alpha)) * np.sin(np.radians(ADD[t]))
        ankle_z = l_hiptoankle * np.sin(np.radians(alpha))
        coo_hip.append(np.array([0, 0, 0]))
        coo_knee.append(np.array([knee_x, knee_y, knee_z]))
        coo_ankle.append(np.array([ankle_x, ankle_y, ankle_z]))
    return np.array(coo_hip), np.array(coo_knee), np.array(coo_ankle)


# PLOTTING FUNCTIONS

# Plot body
def plot_3D_rectangle(length,lar,ax,thick,elevation=0,resolution=10,color="r",x_center=0,y_center=0,order=0):
    x = np.linspace(x_center - length / 2, x_center + length / 2, resolution)
    z = np.linspace(y_center - lar / 2, y_center + lar / 2, resolution)
    X, Z = np.meshgrid(x, z)
    Y = X * 0 + elevation + thick
    ax.plot_surface(X, Z, Y - thick, linewidth=0, color=color, antialiased=True, shade=True)
    face1 = Rectangle((x_center - length / 2, elevation), length, thick, color="#555555")
    ax.add_patch(face1)
    art3d.pathpatch_2d_to_3d(face1, z=lar / 2 + y_center, zdir="y")
    face2 = Rectangle((x_center - length / 2, elevation), length, thick, color="#555555")
    ax.add_patch(face2)
    art3d.pathpatch_2d_to_3d(face2, z=-lar / 2 + y_center, zdir="y")
    face4 = Rectangle((y_center - lar / 2, elevation), lar, thick, color="#505050")
    ax.add_patch(face4)
    art3d.pathpatch_2d_to_3d(face4, z=length / 2 + x_center, zdir="x")
    ax.plot_surface(X, Z, Y, linewidth=0, color=color, antialiased=True, shade=True)


## Plot pillow
def plot_3D_cylinder(radius,height,ax,elevation=0,resolution=10,color="r",x_center=0,y_center=0,order=0):
    x = np.linspace(x_center - radius, x_center + radius, resolution)
    z = np.linspace(elevation, elevation + height, resolution)
    X, Z = np.meshgrid(x, z)
    Y = np.sqrt(radius**2 - (X - x_center) ** 2) + y_center  # Pythagorean theorem

    ax.plot_surface(X, Z, Y, linewidth=0, color=color, antialiased=True, shade=True)
    ax.plot_surface(X, Z, (2 * y_center - Y), linewidth=0, color=color, antialiased=True, shade=True)
    ceiling = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(ceiling)
    art3d.pathpatch_2d_to_3d(ceiling, z=elevation + height, zdir="y")
    ax.plot_surface(
        X,
        Z,
        Y,
        linewidth=0,
        color=color,
        antialiased=False,
        shade=True,
        zorder=order + 1,
    )
    ax.plot_surface(
        X,
        Z,
        (2 * y_center - Y),
        linewidth=0,
        color=color,
        antialiased=True,
        shade=True,
        zorder=order,
    )
    floor = Circle((x_center, y_center), radius, color="#606060")
    ax.add_patch(floor)
    art3d.pathpatch_2d_to_3d(floor, z=elevation, zdir="y")


## Gif function
def animate_func(num):
    ax[3].clear()
    ax[1].clear()
    ax[2].clear()
    ########
    plot_3D_rectangle(
        110,
        50,
        ax[3],
        2.0,
        elevation=0,
        resolution=10,
        color="silver",
        x_center=-5,
        y_center=20,
        order=0,
    )
    ax[3].plot3D(
        [hiplx[num, 0], kneelx[num, 0], anklx[num, 0]],
        [hiplx[num, 1] + 10, kneelx[num, 1] + 10, anklx[num, 1] + 10],
        [hiplx[num, 2], kneelx[num, 2], anklx[num, 2]],
        zorder=3,
        color="#FA525B",
        antialiased=True,
        linewidth=20,
        fillstyle="full",
        solid_capstyle="round",
    )
    ax[3].plot3D(
        [hiplx[num, 0], kneelx[num, 0]],
        [hiplx[num, 1] + 10, kneelx[num, 1] + 10],
        [hiplx[num, 2] - 2, kneelx[num, 2]],
        color="#FA525B",
        antialiased=True,
        zorder=3,
        linewidth=18,
        fillstyle="full",
        solid_capstyle="round",
    )
    plot_3D_cylinder(
        5.5,
        -40,
        ax[3],
        elevation=6,
        resolution=20,
        color="#c3c3c3",
        x_center=17,
        y_center=-2,
        order=0,
    )
    ax[3].plot3D(
        [hiprx[0, 0], kneerx[num, 0], ankrx[num, 0]],
        [hiprx[num, 1] - 8, kneerx[num, 1] - 9, ankrx[num, 1] - 8],
        [hiprx[num, 2] + 1 + 1, kneerx[num, 2] + 1, ankrx[num, 2] + 1],
        color="#19D3C5",
        antialiased=True,
        zorder=1,
        linewidth=20,
        fillstyle="full",
        solid_capstyle="round",
    )
    ax[3].plot_surface(x_, y_, z_, color="#656565", antialiased=True, shade=True)
    ax[3].plot_surface(x, y, z, color="grey", antialiased=True, shade=True)
    ax[3].plot3D(
        [hiprx[0, 0] + 3, kneerx[num, 0]],
        [hiprx[num, 1] - 8 + 1, kneerx[num, 1] - 9 + 1],
        [hiprx[num, 2] - 1 + 1 + 1, kneerx[num, 2] + 1],
        color="#19D3C5",
        antialiased=True,
        zorder=2,
        linewidth=18,
        fillstyle="full",
        solid_capstyle="round",
    )

    # Setting Axes Limits
    ax[3].set_xlim3d([-55, 45])
    ax[3].set_ylim3d([-50, 50])
    ax[3].set_zlim3d([-30, 20])
    ax[3].set_xlabel("x", fontsize="15")
    ax[3].set_ylabel("y", fontsize="15")
    ax[3].set_zlabel("z", fontsize="15")
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_zticks([])
    # Make the panes transparent
    ax[3].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[3].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[3].zaxis.set_pane_color((0.76, 0.76, 0.76, 1.0))
    # Make the grid lines transparent
    ax[3].xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax[3].yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax[3].zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    #######################################################
    # LEG PLOT
    ax[1].set_title("LEFT", fontsize="15", pad=2)
    ax[2].set_title("RIGHT", fontsize="15", pad=2)
    ############################################
    start = int(num - window_width / 2)
    end = int(num + window_width / 2)
    ax[1].plot(
        time[start + shift : end + shift],
        20 * exp_stim[start + shift : end + shift, :] + 130,
        color="#fa525b",
    )
    ax[1].plot([0, 0], [170, -1070], linestyle="dotted", color="#fa525b", linewidth=1)
    for r in range(7):
        true_line = ax[1].plot(
            time[start + shift : end + shift],
            0.95 * trial[start + shift : end + shift, r] - r * 170,
            color="#dbdbdd",
            linewidth=1,
        )
        ax[1].set_xticks(ticks, lab)
        ax[1].set_ylim(-1150, 200)
        ax[1].set_xlim(time[start + shift], time[end + shift])
        ax[1].tick_params("x", labelbottom=False)

    ax[1].tick_params("x", labelbottom=True, pad=-15, direction="in", labelsize="12")
    ax[1].set_yticks((-1020, -850, -680, -510, -340, -170, 0))
    ax[1].set_yticklabels(("Sol", "MG", "TA", "ST", "VLat", "RF", "Il"))
    ax[1].tick_params("y", labelsize="13")
    ax[1].set_xlabel("Time [ms]", fontsize="13")

    #####

    ax[2].plot(
        time[start + shift : end + shift],
        20 * exp_stim[start + shift : end + shift, :] + 130,
        color="#fa525b",
    )
    ax[2].plot([0, 0], [170, -1070], linestyle="dotted", color="#fa525b", linewidth=1)
    for r in range(7):
        true_line = ax[2].plot(
            time[start + shift : end + shift],
            0.95 * trial[start + shift : end + shift, r + 7] - r * 170,
            color="#dbdbdd",
            linewidth=1,
        )
        ax[2].set_xticks(ticks, lab)
        ax[2].set_ylim(-1150, 200)
        ax[2].set_xlim(time[start + shift], time[end + shift])
        ax[2].tick_params("x", labelbottom=False)

    ax[2].tick_params("x", labelbottom=True, pad=-15, direction="in", labelsize="12")
    ax[2].set_yticks([])
    ax[2].set_xlabel("Time [ms]", fontsize="13")


# Plot corresponding electrode
def plot_electrode(ax: matplotlib.axes.Axes, cathodes: int, anodes: List[int]):

    wid = 36
    hei = 80
    image = plt.imread(os.path.abspath("../images/anode.png"))
    # OffsetBox
    image_box = OffsetImage(image, zoom=0.15)
    for x0, y0 in zip(x_anodes, y_anodes):
        ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
        rect3 = matplotlib.patches.Rectangle((x0 - 14, y0 - 40), wid, hei, clip_box=ab, color="#8A8A8C", capstyle="round")
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

    image = plt.imread(os.path.abspath("../images/cathode.png"))
    # OffsetBox
    image_box = OffsetImage(image, zoom=0.15)
    for x0, y0 in zip(x_cathodes, y_cathodes):
        ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
        rect3 = matplotlib.patches.Rectangle((x0 - 14, y0 - 40), 36, 80, clip_box=ab, color="#FA525B", capstyle="round")
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
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


###################################################
# SETTINGS
an = [16]
cath = [2]
# for electrode
from utils import ELECTRODE_POSITIONS_ELEC

with open(os.path.abspath("../images/electrode.png"), "rb") as electrode_file:
    electrode_im = plt.imread(electrode_file)
height, width = electrode_im.shape[0], electrode_im.shape[1]
x_offset = 4
y_offset = 90  # 165
x_anodes, y_anodes = [], []
for anode in an:
    x_anodes.append(ELECTRODE_POSITIONS_ELEC[anode][0] * width + x_offset)
    y_anodes.append(ELECTRODE_POSITIONS_ELEC[anode][1] * height + y_offset)
x_cathodes, y_cathodes = [], []
for cathode in cath:
    x_cathodes.append(ELECTRODE_POSITIONS_ELEC[cathode][0] * width + x_offset)
    y_cathodes.append(ELECTRODE_POSITIONS_ELEC[cathode][1] * height + y_offset)
x_anodes, y_anodes = np.atleast_1d(x_anodes, y_anodes)
x_cathodes, y_cathodes = np.atleast_1d(x_cathodes, y_cathodes)


# Plotting the Animation
shadow_leg = True
if shadow_leg:
    with open("leftlegs466.npy", "rb") as f:
        ang_ = np.load(f, allow_pickle=True)
    ang_ = ang_[0, :, :].T
    hiplx, kneelx, anklx = to_coo(ang_[0] + 30.0, 0.0 * ang_[0], ang_[1] - 57.0)
    hiprx, kneerx, ankrx = to_coo(ang_[2] + 30.0, 0.0 * ang_[0], ang_[3] - 57.0)

numDataPoints = 1752 - 1
duration = int(1000 * numDataPoints / stimfs)
actual_time = np.linspace(0, duration - 1, numDataPoints)
window_width = 200
shift = int(window_width / 2)
time = np.linspace(
    -1000 * shift / stimfs,
    duration - 1 + 1000 * shift / stimfs,
    numDataPoints + window_width,
)
ticks = np.arange(-50, duration - 1 + shift, 50)

lab = [""]
for i in ticks[1:]:
    lab = np.append(lab, str(int(i)))


# Stim processing
stimtrial = stim[0, :, :]
cathode_index = np.argmax(np.max(stimtrial, axis=0))
peaks = find_peaks(stimtrial[:, cathode_index])


# EMG processing
emg_trial = emg[0, :, :]
arr = np.empty((int(window_width / 2), 14))
arr[:] = 0
arr1 = np.zeros((int(window_width / 2), 17))
arr2 = np.ones((int(window_width / 2), 3))

exp_stim = np.concatenate((arr1, stimtrial[shift:], arr1), axis=0)
trial = np.concatenate((arr, emg_trial[shift:], arr), axis=0)  # add nan before and after emg_trial and time
hiplx = np.concatenate((arr2 * hiplx[0], hiplx[shift:], arr2 * hiplx[-1]), axis=0)
anklx = np.concatenate((arr2 * anklx[0], anklx[shift:], arr2 * anklx[-1]), axis=0)
kneelx = np.concatenate((arr2 * kneelx[0], kneelx[shift:], arr2 * kneelx[-1]), axis=0)
hiprx = np.concatenate((arr2 * hiprx[0], hiprx[shift:], arr2 * hiprx[-1]), axis=0)
ankrx = np.concatenate((arr2 * ankrx[0], ankrx[shift:], arr2 * ankrx[-1]), axis=0)
kneerx = np.concatenate((arr2 * kneerx[0], kneerx[shift:], arr2 * kneerx[-1]), axis=0)


r = 9
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:30j, 0.0 : 2.0 * pi : 30j]
x = 1.3 * r * sin(phi) * cos(theta) - 56.5
y = 1.1 * r * sin(phi) * sin(theta) + 1
z = 0.7 * r * cos(phi) + 1 * 2
x_ = np.array([[hiplx[0, 0], hiplx[0, 0], -30, -50],[hiplx[0, 0], hiplx[0, 0], -30, -50],[hiplx[0, 0], hiplx[0, 0], -30, -50],[hiplx[0, 0], hiplx[0, 0], -30, -50]])
y_ = (np.array([[hiplx[0, 1], hiplx[0, 1], hiplx[0, 1] + 3, hiplx[0, 1] + 6],[hiplx[0, 1], hiplx[0, 1], hiplx[0, 1] + 3, hiplx[0, 1] + 6],[hiplx[0, 1] - 22, hiplx[0, 1] - 22, hiplx[0, 1] - 25, hiplx[0, 1] - 28],[hiplx[0, 1] - 22, hiplx[0, 1] - 22, hiplx[0, 1] - 25, hiplx[0, 1] - 28]])+ 13)
z_ = np.array(
    [[0.9 * hiplx[0, 2] - 3,0.9 * hiplx[0, 2] - 3,0.9 * hiplx[0, 2] - 3,0.9 * hiplx[0, 2] - 3],[0.9 * hiplx[0, 2] - 3,1.1 * hiplx[0, 2] + 4,1.1 * hiplx[0, 2] + 4,0.9 * hiplx[0, 2] - 3],[0.9 * hiplx[0, 2] - 3,1.1 * hiplx[0, 2] + 4,1 * hiplx[0, 2] + 4,0.9 * hiplx[0, 2] - 3],[0.9 * hiplx[0, 2] - 3,0.9 * hiplx[0, 2] - 3,0.9 * hiplx[0, 2] - 30.9 * hiplx[0, 2] - 3,]])

###################################################
# Actually plotting
background_color = "#323335"
fig, ax = plt.subplots(
    1,
    4,
    figsize=(15, 10),
    facecolor=background_color,
    frameon=False,
    gridspec_kw={"width_ratios": [2, 5, 5, 7]},
)
ax[3] = plt.subplot(144, projection="3d", computed_zorder=False, facecolor=background_color)
ax[3].view_init(20, 80)
ax[2] = plt.subplot(143, facecolor=background_color)
ax[1] = plt.subplot(142, facecolor=background_color)
ax[0] = plt.subplot(141, facecolor=background_color)
plt.setp(ax[0].spines.values(), color=background_color)
plt.setp(ax[1].spines.values(), color=background_color)
plt.setp(ax[2].spines.values(), color=background_color)

plot_electrode(ax[0], [2], [16])
line_ani = animation.FuncAnimation(fig, animate_func, interval=1.2, frames=numDataPoints)
a = ax[1].get_xticks
plt.tight_layout()
plt.show()
###################################################
# Saving the Animation
f = r"c:/Users/axell/Desktop/final.gif"
writergif = animation.PillowWriter(fps=40)
line_ani.save(f, writer=writergif)
###################################################
