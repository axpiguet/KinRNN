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
import mpl_toolkits.mplot3d.art3d as art3d
###################################################


selected_trial = 54
###################################################

# Load data
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

emg =  normalize(emgold, list(emgold.columns[-14:]), fsemg)
selected_trial = emg.loc[((emg['Cathodes'].astype(str) == '[5]') &(emg['Anodes'].astype(str) == '[4]') & (emg['Frequency'] == 20)& (emg['Amplitude'] == 1.5))].index.values[0]

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


# Functions for plotting

def animate_func(num):
    ax[3].clear()
    ax[1].clear()
    ax[2].clear()

    ax[3].plot3D([rhip[num,0],rknee[num,0],rank[num,0]],[rhip[num,1]-9,rknee[num,1]-9,rank[num,1]-9], [rhip[num,2],rknee[num,2],rank[num,2]], color='cornflowerblue',zorder=1,antialiased=True, linewidth=15,fillstyle='full' , solid_capstyle='round')
    ax[3].plot3D([rhip[num,0],rknee[num,0]],[rhip[num,1]-9,rknee[num,1]-9], [rhip[num,2]-2.5,rknee[num,2]], color='cornflowerblue',antialiased=True, linewidth=15,zorder=2,fillstyle='full', solid_capstyle='round')

    ax[3].plot_surface(x_, y_, z_ , color = '#D0C1D1',antialiased=True,shade=True)
    ax[3].plot_surface(x, y, z,color = '#D0C1D1',antialiased=True,shade=True)


    ax[3].plot3D([lhip[num,0],lknee[num,0],lank[num,0]],[lhip[num,1]+10,lknee[num,1]+10,lank[num,1]+10], [lhip[num,2],lknee[num,2],lank[num,2]], color='#fa525b',antialiased=True, linewidth=15,fillstyle='full' , solid_capstyle='round')
    ax[3].plot3D([lhip[num,0],lknee[num,0]],[lhip[num,1]+10,lknee[num,1]+10], [lhip[num,2]-2.5,lknee[num,2]], color='#fa525b',antialiased=True, linewidth=15,fillstyle='full', solid_capstyle='round')
    # Setting Axes Limits
    ax[3].set_xlim3d([-50, 50])
    ax[3].set_ylim3d([-50, 50])
    ax[3].set_zlim3d([-25,25])

    ax[3].set_title('Trial '+ str(selected_trial), fontsize = "15")
    ax[3].set_xlabel('x', fontsize = "15")
    ax[3].set_ylabel('y', fontsize = "15")
    ax[3].set_zlabel('z', fontsize = "15")
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_zticks([])
    # Make the panes transparent
    ax[3].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[3].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[3].zaxis.set_pane_color((0.32, 0.30, 0.32, 1.0))
    # Make the grid lines transparent
    ax[3].xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax[3].yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax[3].zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    #######################################################
    # LEGs PLOT
    ax[1].set_title('LEFT', fontsize='15', pad = 2)
    ax[2].set_title('RIGHT', fontsize='15', pad = 2)
    ############################################
    start = int(num-window_width/2)
    end = int(num+ window_width/2)
    ax[1].plot(time[start+shift:end+shift], 20*exp_stim[start+shift:end+shift,:]+130, color='#fa525b')
    ax[1].plot([0,0], [170,-1070],linestyle='dotted',  color='#fa525b', linewidth=1)

    for r in range(7):
        true_line = ax[1].plot(time[start+shift:end+shift], trial[start+shift:end+shift,r]-r*170,  color="#dbdbdd", linewidth=1)

        ax[1].set_xticks(ticks, lab)
        ax[1].set_ylim(-1150,200)
        ax[1].set_xlim(time[start+shift],time[end+shift])
        ax[1].tick_params('x', labelbottom=False)

    ax[1].tick_params('x', labelbottom=True, pad = -15, direction = 'in', labelsize="12")
    ax[1].set_yticks((-1020,-850,-680,-510,-340,-170,0))
    ax[1].set_yticklabels(('Sol','TA','MG','ST','VLat','RF','Add'))
    ax[1].tick_params('y',  labelsize="13")
    ax[1].set_xlabel('Time [ms]', fontsize="13")
#####
    ax[2].plot(time[start+shift:end+shift], 20*exp_stim[start+shift:end+shift,:]+130, color='#fa525b')
    ax[2].plot([0,0], [170,-1070],linestyle='dotted',  color='#fa525b', linewidth=1)
    for r in range(7):
        true_line = ax[2].plot(time[start+shift:end+shift], trial[start+shift:end+shift,r+7]-r*170,  color="#dbdbdd", linewidth=1)

        ax[2].set_xticks(ticks, lab)
        ax[2].set_ylim(-1150,200)
        ax[2].set_xlim(time[start+shift],time[end+shift])
        ax[2].tick_params('x', labelbottom=False)


    ax[2].tick_params('x', labelbottom=True, pad = -15, direction = 'in', labelsize="12")
    ax[2].set_yticks([])
    ax[2].set_xlabel('Time [ms]', fontsize="13")



def plot_electrode(ax: matplotlib.axes.Axes, cathodes: int, anodes: List[int]):
    wid = 36
    hei = 80
    image = plt.imread(os.path.abspath( '../images/anode.png'))
    # OffsetBox
    image_box = OffsetImage(image, zoom=0.15)
    for x0, y0 in zip(x_anodes, y_anodes):
        ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
        rect3 = matplotlib.patches.Rectangle((x0-14, y0-40),wid, hei,clip_box = ab, color = '#8A8A8C', joinstyle = 'round')
        pos = (x0-14 + int(wid/2), y0-40+ int(hei/2))

        ax.text(pos[0], pos[1], '+', fontsize = 'xx-large' ,color = 'snow', horizontalalignment = 'center', verticalalignment = 'center')
        ax.add_patch(rect3)

    image = plt.imread(os.path.abspath( '../images/cathode.png'))
    # OffsetBox
    image_box = OffsetImage(image, zoom=0.15)
    for x0, y0 in zip(x_cathodes, y_cathodes):
        ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
        rect3 = matplotlib.patches.Rectangle((x0-14, y0-40),36, 80,clip_box = ab, color = '#FA525B', joinstyle = 'round')
        pos = (x0-14 + int(wid/2), y0-40+ int(hei/2))

        ax.text(pos[0], pos[1], '-', fontsize = 'xx-large' ,color = 'snow',horizontalalignment = 'center', verticalalignment = 'center')
        ax.add_patch(rect3)

    ax.imshow(electrode_im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

###################################################
# SETTINGS

# for electrode
from utils import ELECTRODE_POSITIONS_ELEC
with open(os.path.abspath( '../images/electrode.png'), 'rb') as electrode_file:
    electrode_im = plt.imread(electrode_file)
height, width = electrode_im.shape[0], electrode_im.shape[1]
x_offset = 4
y_offset = 90
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
numDataPoints = int(np.array(list(stim.iloc[selected_trial][stim.columns[-17:]].values)).shape[1])


lhip, lknee, lank = to_coo(kin.LHIPF.iloc[selected_trial], kin.LAdd.iloc[selected_trial],kin.LKNEX.iloc[selected_trial] )
rhip, rknee, rank = to_coo(kin.RHIPF.iloc[selected_trial], kin.RAdd.iloc[selected_trial],kin.RKNEX.iloc[selected_trial] )
duration =  int(1000*lhip.shape[0]/fskin)


fig1,ax1 = plt.subplots(2,2)

x = np.linspace(0, duration-1,lhip.shape[0])

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
fig1.savefig('kin.png')
##############################33

lhipfunc = interpolate.interp1d(x, lhip.T)
lkneefunc = interpolate.interp1d(x, lknee.T)
lankfunc = interpolate.interp1d(x, lank.T)
rhipfunc = interpolate.interp1d(x, rhip.T)
rkneefunc = interpolate.interp1d(x, rknee.T)
rankfunc = interpolate.interp1d(x, rank.T)

actual_time = np.linspace(0, duration-1,numDataPoints)
lhip = lhipfunc(actual_time).T
lank = lankfunc(actual_time).T
lknee = lkneefunc(actual_time).T
rhip = rhipfunc(actual_time).T
rank = rankfunc(actual_time).T
rknee = rkneefunc(actual_time).T


window_width = 200
shift = int(window_width/2)
emg_trial1 = emg.iloc[selected_trial]
pooh= np.array(list(emg_trial1[MUSCLES].values)).T
emgfunc = interpolate.interp1d(np.linspace(0, duration-1,pooh.shape[0]), pooh.T)
emg_trial = emgfunc(actual_time).T
ticks = np.arange(-50,duration-1+shift,50)

lab = ['']
for i in ticks[1:] :
    lab = np.append(lab, str(int(i)))
arr = np.empty((int(window_width/2),emg_trial.shape[1]))
arr[:] = 0

stimtrial = np.array(list(stim.iloc[selected_trial][stim.columns[-17:]].values)).T
fig, ax = plt.subplots(2, 1)
ax[0].plot(np.linspace(0,10,stimtrial.shape[0]),stimtrial)
ax[0].plot(np.linspace(0,10,lknee.shape[0]),lknee[:,1:])
stimfunc = interpolate.interp1d(np.linspace(0, duration-1,stimtrial.shape[0]), stimtrial.T)
stimtrial = stimfunc(actual_time).T


cathode_index = np.argmax(np.max(stimtrial, axis=0))
peaks = find_peaks(stimtrial[:,cathode_index])

begin = peaks[0][0] - 20


ax[1].plot(np.linspace(0,10,stimtrial.shape[0]),15*stimtrial)
ax[1].plot(np.linspace(0,10,emg_trial.shape[0]),emg_trial-70)
plt.savefig('ooo.png')
arr1 = np.zeros((int(window_width/2),17))
arr2 = np.ones((int(window_width/2),3))
arr3 = np.ones((int(window_width/2),1))

LH = np.array([list(jointLHfunc(np.linspace(0, duration-1,numDataPoints)))]).T
LK = np.array([list(jointLKfunc(np.linspace(0, duration-1,numDataPoints)))]).T
LH = np.concatenate(( arr3*LH[0], LH[begin:],arr3*LH[-1] ), axis=0)
LK = np.concatenate(( arr3*LK[0], LK[begin:],arr3*LK[-1] ), axis=0)

exp_stim = np.concatenate((arr1, stimtrial[begin:],arr1), axis=0)
trial =  np.concatenate((arr, emg_trial[begin:],arr), axis=0) # add nan before and after emg_trial and time

lhip = np.concatenate(( arr2*lhip[0], lhip[begin:],arr2*lhip[-1] ), axis=0)
lank = np.concatenate(( arr2*lank[0], lank[begin:],arr2*lank[-1] ), axis=0)
lknee = np.concatenate((  arr2*lknee[0], lknee[begin:],arr2*lknee[-1] ), axis=0)
rhip = np.concatenate((  arr2*rhip[0], rhip[begin:],arr2*rhip[-1] ), axis=0)
rank = np.concatenate((  arr2*rank[0], rank[begin:],arr2*rank[-1] ), axis=0)
rknee =  np.concatenate((  arr2*rknee[0], rknee[begin:],arr2*rknee[-1] ), axis=0)

## PLOTTING
numDataPoints = exp_stim.shape[0]-1
time = np.linspace(-1000*shift/stimfs, duration-1+shift-1000*begin/stimfs,numDataPoints+window_width)


r = 9
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:30j, 0.0:2.0*pi:30j]
x = 1.3*r*sin(phi)*cos(theta)-56.5
y = 1.1*r*sin(phi)*sin(theta)+1
z = 0.7*r*cos(phi)+1*2

x_ = np.array([[lhip[0,0], lhip[0,0], -30,-50],[lhip[0,0], lhip[0,0],  -30,-50], [lhip[0,0], lhip[0,0],  -30,-50], [lhip[0,0], lhip[0,0],  -30,-50]])
y_ = np.array([[lhip[0,1], lhip[0,1], lhip[0,1]+3,lhip[0,1]+6],[lhip[0,1], lhip[0,1],lhip[0,1]+3,lhip[0,1]+6], [lhip[0,1]-22, lhip[0,1]-22,lhip[0,1]-25,lhip[0,1]-28], [lhip[0,1]-22, lhip[0,1]-22, lhip[0,1]-25,lhip[0,1]-28]])+13
z_ = np.array([[0.9*lhip[0,2]-3, 0.9*lhip[0,2]-3, 0.9*lhip[0,2]-3,0.9*lhip[0,2]-3],[0.9*lhip[0,2]-3, 1.1*lhip[0,2]+4, 1.1*lhip[0,2]+4,0.9*lhip[0,2]-3], [0.9*lhip[0,2]-3, 1.1*lhip[0,2]+4, 1*lhip[0,2]+4,0.9*lhip[0,2]-3], [0.9*lhip[0,2]-3, 0.9*lhip[0,2]-3, 0.9*lhip[0,2]-3,0.9*lhip[0,2]-3]])
#
###################################################
# Actually plotting
background_color =  '#323335'
fig, ax = plt.subplots(1, 4 , figsize= (15,10) , facecolor =background_color,frameon = False , gridspec_kw={'width_ratios': [2,5,5,7]})
ax[3] = plt.subplot(144,projection='3d',computed_zorder=False, facecolor = background_color)
ax[3].view_init(20, 75)
ax[2] = plt.subplot(143, facecolor = background_color)
ax[1] = plt.subplot(142, facecolor = background_color)
ax[0] = plt.subplot(141, facecolor = background_color)
plt.setp(ax[0].spines.values(), color=background_color)
plt.setp(ax[1].spines.values(), color=background_color)
plt.setp(ax[2].spines.values(), color=background_color)

# Saving
a = np.vstack((LH.T, LK.T)).T
with open('leftleg.npy', 'wb') as f:
    np.save(f, a,allow_pickle=False)

with open('stimleg.npy', 'wb') as f:
    np.save(f, exp_stim, allow_pickle=False)

with open('emg54.npy', 'wb') as f:
    np.save(f,trial, allow_pickle=False)

plot_electrode(ax[0], kin['Cathodes'].iloc[selected_trial],kin['Anodes'].iloc[selected_trial])
line_ani = animation.FuncAnimation(fig, animate_func, interval=1.2,frames=200)
a = ax[1].get_xticks
plt.tight_layout()
plt.show()
###################################################

f = r"c:/Users/axell/Desktop/truc.gif"

writergif = animation.PillowWriter(fps=40)
line_ani.save(f, writer=writergif)
###################################################
