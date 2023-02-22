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
import math
from matplotlib import animation


def to_coo (HIPF, ADD, KNEX):
    l_thigh = 20; l_shank = 20;
    coo_hip = []
    coo_knee = []
    coo_ankle = []
    for t in range(len(HIPF)):
          knee_x = l_thigh*np.cos(np.radians(HIPF[t]))*np.cos(np.radians(ADD[t]));
          knee_y = l_thigh*np.cos(np.radians(HIPF[t]))*np.sin(np.radians(ADD[t]));
          knee_z = l_thigh*np.sin(np.radians(HIPF[t]));
          l_hiptoankle = np.sqrt(l_thigh**2 + l_shank**2 - 2*l_thigh*l_shank*np.cos(np.radians(180-KNEX[t])));
          #theta = math.acos(np.radians((l_thigh**2 + l_hiptoankle**2 - l_shank**2)/(2*l_thigh*l_hiptoankle)));
          theta = math.degrees(math.acos((l_thigh**2 + l_hiptoankle**2 - l_shank**2)/(2*l_thigh*l_hiptoankle)));
          alpha = HIPF[t] - theta;
          ankle_x = l_hiptoankle*np.cos(np.radians(alpha))*np.cos(np.radians(ADD[t]));
          ankle_y = l_hiptoankle*np.cos(np.radians(alpha))*np.sin(np.radians(ADD[t]));
          ankle_z = l_hiptoankle*np.sin(np.radians(alpha));
          coo_hip.append(np.array([0, 0, 0]));
          coo_knee.append(np.array([knee_x, knee_y, knee_z]));
          coo_ankle.append(np.array([ankle_x, ankle_y, ankle_z]));
    return np.array(coo_hip), np.array(coo_knee) , np.array(coo_ankle)

kin = cPickle.load(open("kinMay21.pkl", "rb" ))
essai = kin.iloc[0]
rhip, rknee, rank = to_coo(essai.RHIPF,essai.RAdd,essai.RKNEX )
lhip, lknee, lank = to_coo(essai.LHIPF,essai.LAdd,essai.LKNEX )



from mpl_toolkits.mplot3d import Axes3D

numDataPoints = lhip.shape[0]-1#len(t)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax = plt.subplot(111,projection='3d',computed_zorder=False)
def animate_func(num):

    ax.clear() # Clears the figure to update the line, point,

    # Updating Trajectory Line (num+1 due to Python indexing)

    #marker_data = np.array(list(df_kinematics.iloc[selected_trial,11:19].values))

    #ax[0].plot3D([RTOE[0][num],RKNE[0][num],RHIP[0][num],LHIP[0][num],LKNE[0][num],LTOE[0][num]], [RTOE[1][num],RKNE[1][num],RHIP[1][num],LHIP[1][num],LKNE[1][num],LTOE[1][num]], [RTOE[2][num],RKNE[2][num],RHIP[2][num],LHIP[2][num],LKNE[2][num],LTOE[2][num]], color='#fa525b', linewidth=5)
    #ax[2].plot3D(marker_data[:,num,0], marker_data[:,num,1], marker_data[:,num,2], color='#fa525b', linewidth=8)

    ###ax[2].plot_surface(x_, y_, z_ , color = '#9D9B9B',antialiased=True,shade=True)
    #head
    ###ax[2].plot_surface(x, y, z,color = '#9D9B9B',antialiased=True,shade=True)
    #ax[2].plot3D(marker_data[:,num,0], marker_data[:,num,1], marker_data[:,num,2], color='#fa525b', linewidth=8)
    ax.plot3D([rhip[num,0],rknee[num,0],rank[num,0]] ,[rhip[num,1]-20,rknee[num,1]-20,rank[num,1]-20], [rhip[num,2],rknee[num,2],rank[num,2]], color='#fa525b',antialiased=True, linewidth=12,fillstyle='full')
    ax.plot3D([lhip[num,0],lknee[num,0],lank[num,0]],[lhip[num,1]+10,lknee[num,1]+10,lank[num,1]+10], [lhip[num,2],lknee[num,2],lank[num,2]], color='#fa525b',antialiased=True, linewidth=12,fillstyle='full')
    #y_ = np.array([[marker_data[3][0,1], marker_data[3][0,1], -0.3*marker_data[3][0,1],-0.5*marker_data[3][0,1]], [marker_data[4][0,1], marker_data[4][0,1], -0.3*marker_data[4][0,1],-0.5*marker_data[4][0,1]]])
    #x_ = np.array([[marker_data[3][0,0], marker_data[3][0,0], 1.15*marker_data[3][0,0],1.3*marker_data[3][0,0]], [marker_data[4][0,0], marker_data[4][0,0], 1.15*marker_data[4][0,0],1.3*marker_data[4][0,0]]])
    #z_ = np.array([[marker_data[3][0,2], 1.15*marker_data[3][0,2], 1.15*marker_data[3][0,2],marker_data[3][0,2]], [marker_data[4][0,2], 1.15*marker_data[4][0,2], 1.15*marker_data[4][0,2],marker_data[4][0,2]]])
    #####
    # Setting Axes Limits
    ax.set_xlim3d([-30, 50])
    ax.set_ylim3d([-30, 30])
    ax.set_zlim3d([-10,10])

    # Adding Figure Labels
    #ax[0].set_title('Trial '+ str(selected_trial) +'\nFrame = ' + str(df_kinematics.iloc[81,0][num]))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # make the panes transparent
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

f = r"c:/Users/axell/Desktop/test.gif"
line_ani = animation.FuncAnimation(fig, animate_func, interval=100,frames=numDataPoints)
writergif = animation.PillowWriter(fps=numDataPoints/10)
line_ani.save(f, writer=writergif)
