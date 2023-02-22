# # CODE FOR GIF


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
import scipy.io
###################################################

# loading
#emg = cPickle.load(open("emg4.pkl", "rb" ))
kin1 = cPickle.load(open("kin4.pkl", "rb" ))

ID = "lstmnewdata123"
train_sets, train_targets = torch.load( f'{data.PATH}/{ID}/train_sets_targets.pt')
testing_sets1, testing_targets1 = torch.load( f'{data.PATH}/{ID}/test_sets_targets.pt')
test_stim_features1 = torch.load( f'{data.PATH}/{ID}/test_stim_features.pt')
pred1 = torch.load( f'{data.PATH}/{ID}/test_pred.pt')

# rematch everything
test_stim_features = pd.DataFrame(columns = test_stim_features1.columns)

targ = []
pred = []
sets = []
kin = kin1.copy()
for i in range(kin1.shape[0]):
    rowfeat = test_stim_features1[(test_stim_features1[['Frequency','Amplitude','PulseWidth','Pulses','Cathodes', 'Anodes']] == pd.Series(list(kin1[['Frequency','Amplitude','PulseWidth','Pulses','Cathodes', 'Anodes']].iloc[i].values), index=['Frequency','Amplitude','PulseWidth','Pulses','Cathodes', 'Anodes'])).all(axis='columns')][test_stim_features.columns]
    if not(rowfeat.empty):
        test_stim_features = test_stim_features.append(rowfeat.iloc[0], ignore_index = False)
        targ.append(testing_targets1[rowfeat.index[0], :, :])
        pred.append(pred1[rowfeat.index[0], :, :])
        sets.append(testing_sets1[rowfeat.index[0], :, :, :])
    else :
        kin = kin.drop(index=i)
testing_sets = np.array(list(sets))
testing_targets = np.array(list(targ))
preds = np.array(list(pred))

###################################################
#??
markers = ['LTOE', 'LANK','LKNE','LHIP','RHIP', 'RKNE', 'RANK', 'RTOE']
cols = ['Trial', 'Frame']
cols.extend(markers)
df_kinematics = pd.DataFrame(columns =cols)
df_kinematics.index = df_kinematics['Trial']


# Setting up Data Set for Animation
numDataPoints = np.array(list(kin.iloc[0,11:19].values)).shape[1]#len(t)
emg = testing_targets

# resample for matching sizes
from sklearn.utils import resample
deletable = []
for i in range (kin.shape[0]):
    if kin.RTOE.iloc[i].shape[0]<3 :
        deletable.append(i)
kin=kin.drop(kin.index[deletable])
test_stim_features=test_stim_features.drop(test_stim_features.index[deletable])
emg = np.delete(emg, deletable, 0)
preds = np.delete(preds, deletable, 0)

df_kinematics = kin
selected_trial = 13
emg_trial = emg[selected_trial].T
pred_trial = preds[selected_trial].T

emg_trial = resample(emg_trial.T, n_samples=kin.RTOE.iloc[selected_trial].shape[0], random_state=0)
pred_trial = resample(pred_trial.T, n_samples=kin.RTOE.iloc[selected_trial].shape[0], random_state=0)
###################################################
# functions for plotting
from utils import plot_electrode_activation
import matplotlib
def animate_func(num):
    ax[2].clear()
    ax[1].clear() # Clears the figure to update the line, point,
    # Updating Trajectory Line (num+1 due to Python indexing)

    #marker_data = np.array(list(df_kinematics.iloc[selected_trial,11:19].values))

    #ax[0].plot3D([RTOE[0][num],RKNE[0][num],RHIP[0][num],LHIP[0][num],LKNE[0][num],LTOE[0][num]], [RTOE[1][num],RKNE[1][num],RHIP[1][num],LHIP[1][num],LKNE[1][num],LTOE[1][num]], [RTOE[2][num],RKNE[2][num],RHIP[2][num],LHIP[2][num],LKNE[2][num],LTOE[2][num]], color='#fa525b', linewidth=5)
    #ax[2].plot3D(marker_data[:,num,0], marker_data[:,num,1], marker_data[:,num,2], color='#fa525b', linewidth=8)

    ax[2].plot_surface(x_, y_, z_ , color = '#9D9B9B',antialiased=True,shade=True)
    #head
    ax[2].plot_surface(x, y, z,color = '#9D9B9B',antialiased=True,shade=True)
    #ax[2].plot3D(marker_data[:,num,0], marker_data[:,num,1], marker_data[:,num,2], color='#fa525b', linewidth=8)
    ax[2].plot3D(marker_data[0:4,num,0], marker_data[0:4,num,1], marker_data[0:4,num,2], color='#fa525b',antialiased=True, linewidth=12,fillstyle='full')
    ax[2].plot3D(marker_data[4:8,num,0], marker_data[4:8,num,1], marker_data[4:8,num,2], color='#fa525b',antialiased=True, linewidth=12,fillstyle='full')
    #y_ = np.array([[marker_data[3][0,1], marker_data[3][0,1], -0.3*marker_data[3][0,1],-0.5*marker_data[3][0,1]], [marker_data[4][0,1], marker_data[4][0,1], -0.3*marker_data[4][0,1],-0.5*marker_data[4][0,1]]])
    #x_ = np.array([[marker_data[3][0,0], marker_data[3][0,0], 1.15*marker_data[3][0,0],1.3*marker_data[3][0,0]], [marker_data[4][0,0], marker_data[4][0,0], 1.15*marker_data[4][0,0],1.3*marker_data[4][0,0]]])
    #z_ = np.array([[marker_data[3][0,2], 1.15*marker_data[3][0,2], 1.15*marker_data[3][0,2],marker_data[3][0,2]], [marker_data[4][0,2], 1.15*marker_data[4][0,2], 1.15*marker_data[4][0,2],marker_data[4][0,2]]])
    #####
    # Setting Axes Limits
    ax[2].set_xlim3d([-700, 700])
    ax[2].set_ylim3d([-1100, 200])
    ax[2].set_zlim3d([0,1300])

    # Adding Figure Labels
    #ax[0].set_title('Trial '+ str(selected_trial) +'\nFrame = ' + str(df_kinematics.iloc[81,0][num]))
    ax[2].set_title('Trial '+ str(selected_trial) +'\nFrame = ' + str(int(df_kinematics.Frame.iloc[selected_trial][num])))
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')
    ax[2].set_zlabel('z')
    # make the panes transparent
    ax[2].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[2].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax[2].xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax[2].yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax[2].zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    #######################################################
    # RIGHT PLOT
    #axs[1].plot(time_stim, inputs[i_config][i_sub_config,:,:], color='#fa525b')
    ax[1].set_frame_on(False)
    ax[1].set_title('LEFT', fontsize='15')
    #ax[1].set_ylim(-100,100)
    #time = np.linspace(0, emg_trial.shape[0],emg_trial.shape[0])

    start = int(num-window_width/2)
    end = int(num+ window_width/2)
    #if start<0 : start = 0
    #if start<0 : end = 0
    #if start>=emg_trial.shape[0]: end = emg_trial.shape[0] -1
    #if start>=emg_trial.shape[0] : start = emg_trial.shape[0]-1
    ax[1].plot(time[start+shift:end+shift], 2*exp_stim[start+shift:end+shift,:]*0.03+2, color='#fa525b')

    for r in range(7):
        #predicted_line = ax[1].plot(time[start:end], emg_trial[start:end,r]*0.5-r*2,  color="#dbdbdd", linewidth=1)
        true_line = ax[1].plot(time[start+shift:end+shift], trial[start+shift:end+shift,r]-r*2,  color="#dbdbdd", linewidth=1)
        predicted_line = ax[1].plot(time[start+shift:end+shift], pred_trial[start+shift:end+shift,r]-r*2,  '--' , color="#dbdbdd", alpha = 0.8 , linewidth=1.8)
        ax[1].plot([0,0], [-13,1],linestyle='dotted',  color='#fa525b', linewidth=0.5)
        #predicted_line = ax[1].plot(time[start:end], essai[start:end,r]*35-r*2, '--' , color="#dbdbdd", linewidth=1) ##53555a
        #predicted_line = ax[1].plot(time[start:end], np.sin(time[start:end])-r*25, '--' , color="#dbdbdd", linewidth=3) ##53555a
        #if labels is not None: expected_line = axs[r,c].plot(time_stim, labels[i_config][i_sub_config,:,c*n_muscles + r-1], '-',color= '#959798', linewidth=4)  #'#313335')
        ax[1].set_frame_on(False)
        ax[1].set_xticks(ticks, lab)
        ax[1].set_ylim(-13,3)
        ax[1].set_xlim(time[start+shift],time[end+shift])
        ax[1].tick_params('x', labelbottom=False)


    ax[1].tick_params('x', labelbottom=True, labelsize="10")
    ax[1].set_yticklabels((' ','Sol','TA','MG','ST','VLat','RF','Add',' '))
    #ax[1].ticklabel_format(axis='x', style='sci')
    ax[1].set_xlabel('Time [ms]', fontsize="10")



def plot_electrode(ax: matplotlib.axes.Axes, cathodes: int, anodes: List[int]):
    #ax.imshow(electrode_im)
    image = plt.imread(os.path.abspath( '../images/anode.png'))
    # OffsetBox
    image_box = OffsetImage(image, zoom=0.15) #0.15
    for x0, y0 in zip(x_anodes, y_anodes):
        ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
        #ax.add_artist(ab)
        #im = ax.imshow(image)
        rect3 = matplotlib.patches.Rectangle((x0-14, y0-40),36, 80,clip_box = ab, color = '#8A8A8C', capstyle = 'round')
        #im.set_clip_path(rect3)
        ax.add_patch(rect3)


    image = plt.imread(os.path.abspath( '../images/cathode.png'))
    # OffsetBox
    image_box = OffsetImage(image, zoom=0.15) #0.15
    for x0, y0 in zip(x_cathodes, y_cathodes):
        ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
        #ax.add_artist(ab)

        #im = ax.imshow(image)
        rect3 = matplotlib.patches.Rectangle((x0-14, y0-40),36, 80,clip_box = ab, color = '#FA525B', capstyle = 'round')
        #im.set_clip_path(rect3)
        ax.add_patch(rect3)

    ax.imshow(electrode_im)
    ax.set_frame_on(False)
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
y_offset = 90 #165
x_anodes, y_anodes = [], []
for anode in df_kinematics['Anodes'].iloc[selected_trial]:
    x_anodes.append(ELECTRODE_POSITIONS_ELEC[anode][0]*width+x_offset)
    y_anodes.append(ELECTRODE_POSITIONS_ELEC[anode][1]*height+y_offset)
x_cathodes, y_cathodes = [], []
for cathode in df_kinematics['Cathodes'].iloc[selected_trial]:
    x_cathodes.append(ELECTRODE_POSITIONS_ELEC[cathode][0]*width+x_offset)
    y_cathodes.append(ELECTRODE_POSITIONS_ELEC[cathode][1]*height+y_offset)
x_anodes, y_anodes = np.atleast_1d(x_anodes, y_anodes)
x_cathodes, y_cathodes = np.atleast_1d(x_cathodes, y_cathodes)
##
window_width = 30
shift = int(window_width/2)
ticks = np.linspace(-shift, emg_trial.shape[0]-1+shift,int((emg_trial.shape[0]+window_width)/5)+1)
lab = ['','','']
for i in ticks[3:] :
    lab = np.append(lab, str(int(i)))
time = np.linspace(-shift, emg_trial.shape[0]-1+shift,emg_trial.shape[0]+window_width)
arr = np.empty((int(window_width/2),13))
arr[:] = 0#np.NaN

stim_duration = 397#emg_trial.shape[0]
stim_arrays = data.create(test_stim_features, stim_duration, fs=229)#data.FS)
print(stim_arrays.shape)
stim = stim_arrays[selected_trial,0,:,:]
#stim =  resample(testing_sets[selected_trial][0,:,:], n_samples=kin.RTOE.iloc[selected_trial].shape[0], random_state=0) # this is for prediction, not plotting
arr1 = np.empty((int(window_width/2),17))
exp_stim = np.concatenate((arr1, stim,arr1), axis=0)
trial =  np.concatenate((arr, emg_trial,arr), axis=0) # add nan before and after emg_trial and time
pred_trial =  np.concatenate((arr, pred_trial,arr), axis=0)
#time = np.concatenate((arr[:,0], time,arr[:,0]), axis=0)
# Plotting the Animation
marker_data = np.array(list(df_kinematics.iloc[selected_trial,11:19].values))
y_ = np.array([[marker_data[3][0,1], marker_data[3][0,1], -0.1*marker_data[3][0,1],-0.7*marker_data[3][0,1]],[marker_data[3][0,1], marker_data[3][0,1], -0.1*marker_data[3][0,1],-0.7*marker_data[3][0,1]], [marker_data[4][0,1], marker_data[4][0,1], -0.3*marker_data[4][0,1],-0.7*marker_data[4][0,1]], [marker_data[4][0,1], marker_data[4][0,1], -0.3*marker_data[4][0,1],-0.7*marker_data[4][0,1]]])
x_ = np.array([[marker_data[3][0,0], marker_data[3][0,0], 1.4*marker_data[3][0,0],1.5*marker_data[3][0,0]],[marker_data[3][0,0], marker_data[3][0,0], 1.3*marker_data[3][0,0],1.5*marker_data[3][0,0]], [marker_data[4][0,0], marker_data[4][0,0], 1.3*marker_data[4][0,0],1.5*marker_data[4][0,0]], [marker_data[4][0,0], marker_data[4][0,0], 1.4*marker_data[4][0,0],1.5*marker_data[4][0,0]]])
#x_ = 1.5*np.array([[-600,-600,-600,-600],[-600,-600,-600,-600], [-100,-100,-100,-100], [-100,-100,-100,-100]])-150
z_ = np.array([[0.9*marker_data[3][0,2], 0.9*marker_data[3][0,2], 0.9*marker_data[3][0,2],0.9*marker_data[3][0,2]],[0.9*marker_data[3][0,2], 1.1*marker_data[3][0,2], 1.1*marker_data[3][0,2],0.9*marker_data[3][0,2]], [0.9*marker_data[4][0,2], 1.1*marker_data[4][0,2], 1*marker_data[4][0,2],0.9*marker_data[4][0,2]], [0.9*marker_data[4][0,2], 0.9*marker_data[4][0,2], 0.9*marker_data[4][0,2],0.9*marker_data[4][0,2]]])
# head
r = 185
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
x = 0.9*r*sin(phi)*cos(theta)+(marker_data[3][0,0]+marker_data[4][0,0])/2
y = 1.1*r*sin(phi)*sin(theta)-marker_data[4][0,1]
z = 1.1*r*cos(phi)+1*marker_data[3][0,2]
#
###################################################
# Actually plotting
fig, ax = plt.subplots(1, 3 , figsize= (13,7),gridspec_kw={'width_ratios': [2,4,5]})
ax[2] = plt.subplot(133,projection='3d',computed_zorder=False)
ax[1] = plt.subplot(132)
ax[0] = plt.subplot(131)
#plot_electrode_activation(ax[2], [4],[1,2])
plot_electrode(ax[0], df_kinematics['Cathodes'].iloc[selected_trial],df_kinematics['Anodes'].iloc[selected_trial])
line_ani = animation.FuncAnimation(fig, animate_func, interval=100,frames=numDataPoints)
a = ax[1].get_xticks
plt.show()
###################################################

###################################################
# Saving the Animation
#f = r"c://Users/axell/Desktop/animate_func.gif"
f = r"c://Users/yes/Desktop/animate_func.gif"
writergif = animation.PillowWriter(fps=numDataPoints/10)
line_ani.save(f, writer=writergif)
###################################################
