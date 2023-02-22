import numpy as np
import _pickle as cPickle

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

from data import MUSCLES, FS, PATH
from utils import plot_electrode_activation


def plot_result(input: np.ndarray, pred: np.ndarray, input_name: dict, ID: str):
    """
    apply low-pass butterworth filter on filtered emg array to get its envelope

    :param inputs: inputs of the network of size (config, time, n_sub_config, n_electrodes)
    :param labels: labels of the inputs of size (congig, time, n_sub_config, n_muscles)
    :param pred: prediction of the net when giving the inputs of size (congig, time, n_sub_config, n_muscles)
    :param input_names: dictionnary with key name, sub_names, cathode and anodes
    :param epoch: actual epoch
    :param training_config: sub training or main training 
    """  
    time_stim = np.linspace(0, input.shape[1]*1000//FS-1, num=input.shape[1])
    n_muscles = len(MUSCLES)//2

    fig = plt.figure(figsize=(40,15)) # 30 15 
    spec = gridspec.GridSpec(pred.shape[2]//2+pred.shape[2]%2+1,3, width_ratios=[0.1,0.45,0.45])
    axs=[]
    for i in range(len(MUSCLES)//2+len(MUSCLES)%2+1):
        raw = []
        for j in range(1,3):
            raw.append(fig.add_subplot(spec[i, j]))
        axs.append(raw)
    axs=np.array(axs)

    axs[0,0].set_ylabel('stim', fontsize="40")
    axs[0,0].tick_params('y',labelsize="30")
    axs[0,1].tick_params('y', labelleft=False)
    axs[0,0].tick_params('x', labelbottom=False, bottom=False)
    axs[0,1].tick_params('x', labelbottom=False, bottom=False)
    axs[0,0].ticklabel_format(axis='y', style='sci')

    for c in range(2):
        axs[0,c].plot(time_stim, input[0,:,:], color='#fa525b')
        axs[0,c].set_frame_on(False)
        axs[0,c].set_title('RIGHT' if c else 'LEFT', fontsize='50')
        axs[0,c].set_ylim(-100,100)

        for r in range(1,len(MUSCLES)//2+len(MUSCLES)%2+1): 
            predicted_line = axs[r,c].plot(time_stim, pred[:,0,c*n_muscles + r-1], '--' , color="#dbdbdd", linewidth=4) ##53555a
            axs[r,c].set_frame_on(False)
            axs[r,c].set_ylim(-100,100)
            axs[r,c].tick_params('x', labelbottom=False)

        axs[-1,c].tick_params('x', labelbottom=True, labelsize="30")
        axs[-1,c].ticklabel_format(axis='x', style='sci')

        axs[-1,c].set_xlabel('time (ms)', fontsize="40")
    
    for r in range(1,len(MUSCLES)//2+len(MUSCLES)%2+1):
        axs[r,0].set_ylabel(f'{MUSCLES[r-1][1:]}', fontsize="40")
        axs[r,0].tick_params('y',labelsize="30")
        axs[r,1].tick_params('y', labelleft=False)
        axs[r,0].ticklabel_format(axis='y', style='sci')
    
    ax = fig.add_subplot(spec[:,0])
    plot_electrode_activation(ax, input_name['cathodes'], input_name['anodes'])
    #plt.legend(handles=[predicted_line, expected_line], labels=['Predicted', 'Expected'], frameon=False, fontsize='xx-large')
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0, right=0.985, hspace=0.9, wspace=0.1)
    plt.savefig(f"{PATH}/controller/{ID}_optimized_stim.png", transparent=True)
    plt.close(fig)


def plot_minimization(metric_values: np.ndarray, muscle:str, ID:str):
    fig = plt.figure()
    plt.plot(np.arange(len(metric_values)), -metric_values, 'o-')
    plt.xlabel('Number of iterations')
    plt.ylabel(f'{muscle} SI')
    plt.box(False)
    plt.savefig(f"{PATH}/controller/{ID} SI minimization.png", transparent=True)
    plt.close(fig)
