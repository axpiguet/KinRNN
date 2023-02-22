import _pickle as cPickle
import math as m
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import matplotlib.axes

from sklearn import linear_model

from .emg import load
import data
import utils
from utils import plot_electrode_activation


NORM = 'mean'
augment=False
PATH = f"/media/marion/PARTAGE/Documents/NeuroRestore/Data_backup/mai2021_march2022/mai2021_march2022_{NORM}norm/{'Augmented/' if augment else 'Non-augmented/'}"

plt.style.use('dark_background')
matplotlib.use('Agg')

CMAP_BAR = ['#372367','#41297a','#4b308c','#55369f','#5f3cb1','#6a46c0','#7958c6','#886bcd','#977dd3','#a690d9','#b4a2e0','#c3b5e6','#d2c7ec','#e1daf2','#f0ecf9']
CMAP_BAR_SI = ['#ff7204','#ff7c1d','#ff852d','#ff8e3c','#ff9649','#ff9f56','#ffa764','#ffaf71','#ffb77e','#ffbf8b','#ffc799','#ffcea7','#ffd6b4','#ffddc3','#ffe5d1']
CMAP_COLORMESH = ListedColormap(['#FA525B','#EB4E5B','#DB4A5C','#CC475D','#BE435E','#AF3F5F','#A03C60','#913861','#823562','#733163','#642D64','#552A65','#452666','#372367', '#34306E', '#323E75','#304B7C','#2D5983','#2B668B','#297492','#268199','#248FA0','#229CA8','#1FAAAF','#1DB7B6','#1BC5BD','#19D3C5','#29D3B6','#3AD4A8','#4AD59A','#5BD68C','#6BD77E','#7CD870','#8CD961','#9DDA53','#ADDB45','#BEDC37'][::-1]) #,matplotlib.colors.to_rgba('w',0)
CMAP_MODULATION = ['#ffc9c5','#ff9894','#fe7171','#fa525b','#f32b34','#dd131a','#ae1418','#c6f1eb','#9de8de','#6dded2','#19d3c5','#1ab3a8','#19958c','#177871']
#CMAP_MODULATION = ['#ffc9c5','#ff9894','#fe7171','#fa525b','#f32b34','#dd131a','#c6f1eb','#9de8de','#6dded2','#19d3c5','#1ab3a8','#19958c']
CMAP_EMGS = ['#55369f','#2D5983','#19D3C5','#B8E600','#F0DF0D','#FFB11E','#FA525B','#a690d9','#9cbddd','#9df3ed','#e9ff8f','#faf39d','#ffe0a5','#fdbabd']
#CMAP_EMGS = ['#55369f','#2D5983','#19D3C5','#B8E600','#FFB11E','#FA525B','#a690d9','#9cbddd','#9df3ed','#e9ff8f','#ffe0a5','#fdbabd']
IMPULSE_RED = '#fa525b'


def plot_amp_stimvsresponse(stim_features, emg_array, amplitudes_range, label_stim_features=None, label_emg_array=None, norm='mean', other_path=None):
    actual_amplitudes = np.unique(stim_features['Amplitude'])
    if label_stim_features is not None : label_actual_amplitudes = np.unique(label_stim_features['Amplitude'])
    for i_amp, amplitude in enumerate(amplitudes_range):
        if amplitude not in actual_amplitudes:
            emg_array = np.insert(emg_array, i_amp, np.nan, axis=0)
        if label_stim_features is not None and amplitude not in label_actual_amplitudes:
            label_emg_array = np.insert(label_emg_array, i_amp, np.nan, axis=0)

    muscle_perc = 100*np.nanmean(np.abs(emg_array), axis=1) if norm=='mean' else 100*np.nanmax(np.abs(emg_array), axis=1)
    if label_stim_features is not None: label_muscle_perc = 100*np.nanmean(np.abs(label_emg_array), axis=1) if norm=='mean' else 100*np.nanmax(np.abs(label_emg_array), axis=1)

    fig = plt.figure(figsize=(30,10))
    spec = gridspec.GridSpec(m.ceil(len(data.MUSCLES)/4),1+4+1, width_ratios=[0.166,0.167,0.167,0.167,0.167,0.166])
    axs=[]
    for i in range(m.ceil(len(data.MUSCLES)/4)):
        raw = []
        raw.append(fig.add_subplot(spec[i, 1]))
        for j in range(2,5):
            raw.append(fig.add_subplot(spec[i, j], sharex=raw[0], sharey=raw[0]))
        axs.append(raw)
    axs=np.array(axs)

    for i in range(m.ceil(len(data.MUSCLES)/4)):
        for j in range(4):
            axs[i,j].set_frame_on(False)
            if (i*(len(data.MUSCLES)//4-1) + j//2 < len(data.MUSCLES)//2):
                if label_emg_array is not None:
                    axs[i,j].plot(amplitudes_range, label_muscle_perc[:,(i*(len(data.MUSCLES)//4-1) + j//2 + (len(data.MUSCLES)//2 + len(data.MUSCLES)%2)*(j%2))], 'X', color='darkgrey', markersize=10)
                axs[i,j].plot(amplitudes_range,muscle_perc[:,(i*(len(data.MUSCLES)//4-1) + j//2 + (len(data.MUSCLES)//2 + len(data.MUSCLES)%2)*(j%2))], '-', color=IMPULSE_RED, linewidth=4)
                axs[i,j].set_title(f'{data.MUSCLES[(i*(len(data.MUSCLES)//4-1) + j//2 + (len(data.MUSCLES)//2 + len(data.MUSCLES)%2)*(j%2))]}', fontsize=27, loc='right',color = 'k')
                axs[-1,j].set_xlabel('Amplitude of stimulation (mA)', fontsize='xx-large',color = 'k')
            else:
                axs[i,j].tick_params(
                axis='both',
                which='both',
                left=False,
                bottom=False,
                color = 'k'
                )
            axs[i,j].tick_params(
            axis='both',
            which='both',
            labelleft=False,
            labelbottom=False,
            color = 'k'
            )
            axs[-1,j].tick_params(labelbottom=True, labelsize='x-large' , color = 'k')
            axs[i,j].set_xlim(min(amplitudes_range), max(amplitudes_range))
            axs[i,j].set_ylim(0, 100)
        axs[i,0].set_ylabel('Muscle activity (%)', fontsize='xx-large',color = 'k')
        axs[i,0].tick_params(labelleft=True, labelsize='x-large', color = 'k')

    cathodes = stim_features['Cathodes'].iloc[0]
    anodes = stim_features["Anodes"].iloc[0]
    ax = fig.add_subplot(spec[:,0])
    plot_electrode_activation(ax,cathodes, anodes)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92,bottom=0.1,left=0.065,right=0.935,hspace=0.24,wspace=0.27)
    name = f"cath{'_'.join(map(str,stim_features['Cathodes'].iloc[0]))}_an{'_'.join(map(str,stim_features['Anodes'].iloc[0]))}_freq{stim_features['Frequency'].iloc[0]}"
    if other_path is None:
        print(PATH)
        plt.savefig(PATH + f'EMGs_images/Amplitude_relationship/{name}.svg', transparent=True,  format='svg', dpi=1200)
    else:
        print(other_path)
        plt.savefig(other_path + f'/{name}.svg', transparent=True,  format='svg', dpi=1200)
    plt.close(fig)



def plot_cmap(ax: matplotlib.axes.Axes,amplitudes, cmap, min_amp=0, max_amp=7):
    i_max_amp = np.where(amplitudes==max_amp)[0][0]
    i_min_amp = np.where(amplitudes==min_amp)[0][0]

    gradient = np.linspace(0, 1, len(amplitudes[i_min_amp:i_max_amp+1]))
    gradient = np.vstack((gradient, gradient))

    ax.imshow(gradient, aspect='auto', cmap=ListedColormap(cmap[i_min_amp:i_max_amp+1]))
    ax.tick_params(axis='y', which='both',left=False, labelleft=False)
    ax.set_xticks(np.linspace(0,len(amplitudes[i_min_amp:i_max_amp+1])-1,len(amplitudes[i_min_amp:i_max_amp+1])),labels=amplitudes[i_min_amp:i_max_amp+1], fontsize='xx-large')
    ax.set_xlabel('Amplitude (mA)', fontsize='xx-large')
    ax.set_frame_on(False)



def plot_bars(stim_features, emg_array, frequencies, amplitudes, norm='mean', other_path=None):

    actual_amplitudes = np.unique(stim_features['Amplitude'])
    n_amplitudes = len(actual_amplitudes)
    y_pos = np.arange(len(frequencies))

    emg_array_copy = np.full((len(frequencies)*n_amplitudes, emg_array.shape[1], emg_array.shape[2]),np.nan)
    for i, index in enumerate(stim_features.index):
        i_freq = np.where(frequencies == stim_features.loc[index,'Frequency'])[0][0]
        i_amp = np.where(actual_amplitudes == stim_features.loc[index,'Amplitude'])[0][0]
        emg_array_copy[i_freq*n_amplitudes + i_amp,:,:] = emg_array[i,:,:]
    emg_array = emg_array_copy

    print(np.shape(emg_array[:,:,:len(data.MUSCLES)//2]))
    if norm=='mean':
        muscle_perc_left = 100*np.nanmean(np.abs(emg_array[:,:,:len(data.MUSCLES)//2]), axis=1)
        muscle_perc_right = 100*np.nanmean(np.abs(emg_array[:,:,len(data.MUSCLES)//2:]), axis=1)
    elif norm=='max':
        muscle_perc_left = 100*np.nanmax(np.abs(emg_array[:,:,:len(data.MUSCLES)//2]), axis=1)
        muscle_perc_right = 100*np.nanmax(np.abs(emg_array[:,:,len(data.MUSCLES)//2:]), axis=1)

    fig = plt.figure(figsize=(20,15))
    muscle_ratio = 1/(len(data.MUSCLES)//2) - 0.1/(len(data.MUSCLES)//2)
    muscle_ratios = [muscle_ratio for i in range(len(data.MUSCLES)//2)]
    muscle_ratios.append(0.07)
    muscle_ratios.append(0.03)
    spec = gridspec.GridSpec(len(data.MUSCLES)//2+2,3, width_ratios=[0.17,0.415,0.415], height_ratios=muscle_ratios)
    axs=[]
    for i in range(len(data.MUSCLES)//2):
        raw = []
        for j in range(1,3):
            raw.append(fig.add_subplot(spec[i, j]))
        axs.append(raw)
    axs=np.array(axs)

    for i_muscle in range(len(data.MUSCLES)//2):
        for direction in [0,1]:
            axs[i_muscle,direction].set_frame_on(False)
            # create a subplot from grid
            for i_amp, amplitude in enumerate(actual_amplitudes):
                i_actual_amp = np.where(amplitudes == amplitude)[0][0]
                if direction == 0:
                    bar_plot_left = axs[i_muscle,direction].barh(y_pos, muscle_perc_left[i_amp::n_amplitudes,i_muscle],color=CMAP_BAR[i_actual_amp], zorder=len(actual_amplitudes)-i_amp)
                    #axs[i_muscle,direction].bar_label(bar_plot_left, fmt='%.1f', fontsize='xx-small', padding=-10)

                else:
                    bar_plot_right = axs[i_muscle,direction].barh(y_pos, muscle_perc_right[i_amp::n_amplitudes,i_muscle], color=CMAP_BAR[i_actual_amp],zorder=len(actual_amplitudes)-i_amp) #label=f"{amplitude} mA")
                    #axs[i_muscle,direction].bar_label(bar_plot_right, fmt='%.1f',  fontsize='xx-small') # padding=-18

            axs[i_muscle,direction].set_yticks(y_pos, labels=frequencies, ha='right' if direction==0 else 'left', x=1.04 if direction==0 else -.04,fontsize='15')
            axs[i_muscle,direction].tick_params(
                axis='y',
                which='both',
                left=False,
                right=False,
                labelleft=direction==1,
                labelright=direction==0)

            axs[i_muscle,direction].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=(i_muscle==len(data.MUSCLES)//2-1),         # ticks along the bottom edge are off
                labelbottom=(i_muscle==len(data.MUSCLES)//2-1),
                labelsize='xx-large',
                width=1,
                length=2)


    for i_muscle in range(len(data.MUSCLES)//2):
        for direction in [0,1]:
            axs[i_muscle, direction].set_xlim(0,100)
        axs[i_muscle,0].invert_xaxis()
        axs[i_muscle,1].set_ylabel(data.MUSCLES[i_muscle][1:], fontsize=25, labelpad=12)
    axs[0,0].set_title('LEFT', loc='right', fontsize=30, pad=12)
    axs[0,1].set_title('RIGHT', loc='left', fontsize=30, pad=12)
    axs[-1,1].set_xlabel("Muscle activity (%)", loc='right', fontsize='xx-large')

    cathodes = stim_features['Cathodes'].iloc[0]
    anodes = stim_features["Anodes"].iloc[0]
    ax = fig.add_subplot(spec[:-2,0])
    plot_electrode_activation(ax,cathodes, anodes)

    ax = fig.add_subplot(spec[-2,:])
    ax.set_visible(False)
    ax = fig.add_subplot(spec[-1,:])
    plot_cmap(ax, amplitudes, CMAP_BAR, min_amp=np.min(actual_amplitudes), max_amp=np.max(actual_amplitudes))

    plt.tight_layout()
    plt.subplots_adjust(top=0.87,bottom=0.14,left=0.02,right=0.98,hspace=0.025,wspace=0.25)
    name = f"cath{'_'.join(map(str,stim_features['Cathodes'].iloc[0]))}_an{'_'.join(map(str,stim_features['Anodes'].iloc[0]))}"
    plt.suptitle(f"{name}", fontsize='35')
    if other_path is None:
        plt.savefig(PATH + f'\EMGs_images\Bars\{name}.png', transparent=True)
    else:
        plt.savefig(other_path + f'\{name}.png', transparent=True)
    print('done bro')

def recrutement(stim_features, emg_array, frequencies, norm='mean'):

    actual_amplitudes = np.unique(stim_features['Amplitude'])
    n_amplitudes = len(actual_amplitudes)

    emg_array_copy = np.full((len(frequencies)*n_amplitudes, emg_array.shape[1], emg_array.shape[2]),np.nan)
    for i, index in enumerate(stim_features.index):
        i_freq = np.where(frequencies == stim_features.loc[index,'Frequency'])[0][0]
        i_amp = np.where(actual_amplitudes == stim_features.loc[index,'Amplitude'])[0][0]
        emg_array_copy[i_freq*n_amplitudes + i_amp,:,:] = emg_array[i,:,:]
    emg_array = emg_array_copy

    if norm=='mean':
        muscle_perc_left = 100*np.nanmean(np.abs(emg_array[:,:,:len(data.MUSCLES)//2]), axis=1)
        muscle_perc_right = 100*np.nanmean(np.abs(emg_array[:,:,len(data.MUSCLES)//2:]), axis=1)
    elif norm=='max':
        muscle_perc_left = 100*np.nanmax(np.abs(emg_array[:,:,:len(data.MUSCLES)//2]), axis=1)
        muscle_perc_right = 100*np.nanmax(np.abs(emg_array[:,:,len(data.MUSCLES)//2:]), axis=1)

    # Left muscles : muscle_perc_left[amplitudes,muscles] = values for all freq
    # Right muscles : muscle_perc_right[amplitudes,muscles] = values for all freq
    return np.concatenate((muscle_perc_left,muscle_perc_right), axis=1),actual_amplitudes

def plot_si_bars(stim_features, si_array, frequencies, amplitudes, norm='mean', other_path=None):

    actual_amplitudes = np.unique(stim_features['Amplitude'])
    n_amplitudes = len(actual_amplitudes)
    y_pos = np.arange(len(frequencies))

    si_array_copy = np.full((len(frequencies)*n_amplitudes, si_array.shape[1]),np.nan)
    for i, index in enumerate(stim_features.index):
        i_freq = np.where(frequencies == stim_features.loc[index,'Frequency'])[0][0]
        i_amp = np.where(actual_amplitudes == stim_features.loc[index,'Amplitude'])[0][0]
        si_array_copy[i_freq*n_amplitudes + i_amp,:] = si_array[i,:]
    si_array = si_array_copy


    fig = plt.figure(figsize=(20,15))
    muscle_ratio = 1/(len(data.MUSCLES)//2) - 0.1/(len(data.MUSCLES)//2)
    muscle_ratios = [muscle_ratio for i in range(len(data.MUSCLES)//2)]
    muscle_ratios.append(0.07)
    muscle_ratios.append(0.03)
    spec = gridspec.GridSpec(len(data.MUSCLES)//2+2,3, width_ratios=[0.17,0.415,0.415], height_ratios=muscle_ratios)
    axs=[]
    for i in range(len(data.MUSCLES)//2):
        raw = []
        for j in range(1,3):
            raw.append(fig.add_subplot(spec[i, j]))
        axs.append(raw)
    axs=np.array(axs)

    for i_muscle in range(len(data.MUSCLES)//2):
        for direction in [0,1]:
            axs[i_muscle,direction].set_frame_on(False)
            # create a subplot from grid
            for i_amp, amplitude in enumerate(actual_amplitudes):
                i_actual_amp = np.where(amplitudes == amplitude)[0][0]
                if direction == 0:
                    bar_plot_left = axs[i_muscle,direction].barh(y_pos, si_array[i_amp::n_amplitudes,i_muscle],color=CMAP_BAR_SI[i_actual_amp], zorder=len(actual_amplitudes)-i_amp)
                    #axs[i_muscle,direction].bar_label(bar_plot_left, fmt='%.1f', fontsize='xx-small', padding=-10)

                else:
                    bar_plot_right = axs[i_muscle,direction].barh(y_pos, si_array[i_amp::n_amplitudes,i_muscle], color=CMAP_BAR_SI[i_actual_amp],zorder=len(actual_amplitudes)-i_amp) #label=f"{amplitude} mA")
                    #axs[i_muscle,direction].bar_label(bar_plot_right, fmt='%.1f',  fontsize='xx-small') # padding=-18

            axs[i_muscle,direction].set_yticks(y_pos, labels=frequencies, ha='right' if direction==0 else 'left', x=1.04 if direction==0 else -.04,fontsize='15')
            axs[i_muscle,direction].tick_params(
                axis='y',
                which='both',
                left=False,
                right=False,
                labelleft=direction==1,
                labelright=direction==0)

            axs[i_muscle,direction].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=(i_muscle==len(data.MUSCLES)//2-1),         # ticks along the bottom edge are off
                labelbottom=(i_muscle==len(data.MUSCLES)//2-1),
                labelsize='xx-large',
                width=1,
                length=2)


    for i_muscle in range(len(data.MUSCLES)//2):
        for direction in [0,1]:
            axs[i_muscle, direction].set_xlim(0,1)
        axs[i_muscle,0].invert_xaxis()
        axs[i_muscle,1].set_ylabel(data.MUSCLES[i_muscle][1:], fontsize=25, labelpad=12)
    axs[0,0].set_title('LEFT', loc='right', fontsize=30, pad=12)
    axs[0,1].set_title('RIGHT', loc='left', fontsize=30, pad=12)
    axs[-1,1].set_xlabel("Selectivity index", loc='right', fontsize='xx-large')

    cathodes = stim_features['Cathodes'].iloc[0]
    anodes = stim_features["Anodes"].iloc[0]
    ax = fig.add_subplot(spec[:-2,0])
    plot_electrode_activation(ax,cathodes, anodes)

    ax = fig.add_subplot(spec[-2,:])
    ax.set_visible(False)
    ax = fig.add_subplot(spec[-1,:])
    plot_cmap(ax, amplitudes, CMAP_BAR_SI, min_amp=np.min(actual_amplitudes), max_amp=np.max(actual_amplitudes))

    plt.tight_layout()
    plt.subplots_adjust(top=0.87,bottom=0.14,left=0.02,right=0.98,hspace=0.025,wspace=0.25)
    name = f"cath{'_'.join(map(str,stim_features['Cathodes'].iloc[0]))}_an{'_'.join(map(str,stim_features['Anodes'].iloc[0]))}"
    plt.suptitle(f"{name}", fontsize='35')
    if other_path is None:
        plt.savefig(PATH + f'EMGs_images/SI/{name}.png', transparent=True)
    else:
        plt.savefig(other_path + f'/{name}.png', transparent=True)
    plt.close(fig)


def plot_colormesh(stim_features, emg_array, amplitudes, frequencies, norm='mean', other_path=None):

        actual_frequencies = np.unique(stim_features['Frequency'])
        n_frequencies = len(actual_frequencies)
        actual_amplitudes = np.unique(stim_features['Amplitude'])
        n_amplitudes = len(actual_amplitudes)

        for i_freq, frequency in enumerate(actual_frequencies):
            freq_df = np.array(stim_features.loc[stim_features['Frequency'] == frequency]['Amplitude'])
            for i_amp, amplitude in enumerate(actual_amplitudes):
                if amplitude not in freq_df:
                    emg_array = np.insert(emg_array, i_freq*n_amplitudes + i_amp, np.nan, axis=0)

        muscle_perc = 100*np.nanmean(np.abs(emg_array), axis=1) if norm=='mean' else 100*np.nanmax(np.abs(emg_array), axis=1)
        muscle_perc = muscle_perc.reshape((n_frequencies, n_amplitudes,len(data.MUSCLES)), order='C')

        fig = plt.figure(figsize=(30,10))
        spec = gridspec.GridSpec(m.ceil(len(data.MUSCLES)/4),1+4+1, width_ratios=[0.166,0.167,0.167,0.167,0.167,0.166])
        axs=[]
        for i in range(m.ceil(len(data.MUSCLES)/4)):
            raw = []
            raw.append(fig.add_subplot(spec[i, 1]))
            for j in range(2,5):
                raw.append(fig.add_subplot(spec[i, j], sharex=raw[0], sharey=raw[0]))
            axs.append(raw)
        axs=np.array(axs)
        x = actual_amplitudes
        y = actual_frequencies

        for i in range(m.ceil(len(data.MUSCLES)/4)):
            for j in range(4):
                axs[i,j].set_frame_on(False)
                if (i*(len(data.MUSCLES)//4-1) + j//2 < len(data.MUSCLES)//2):
                    im = axs[i,j].pcolormesh(x,y,muscle_perc[:,:,(i*(len(data.MUSCLES)//4-1) + j//2 + (len(data.MUSCLES)//2 + len(data.MUSCLES)%2)*(j%2))], cmap=CMAP_COLORMESH, vmin=0, vmax=100, shading='nearest')#np.max(muscle_perc))
                    axs[i,j].set_title(f'{data.MUSCLES[(i*(len(data.MUSCLES)//4-1) + j//2 + (len(data.MUSCLES)//2 + len(data.MUSCLES)%2)*(j%2))]}', fontsize='large', loc='right')
                    axs[-1,j].set_xlabel('Amplitude (mA)', fontsize='medium')
                else:
                    axs[i,j].tick_params(
                    axis='both',
                    which='both',
                    left=False,
                    bottom=False
                    )
                axs[i,j].tick_params(
                axis='both',
                which='both',
                labelleft=False,
                labelbottom=False
                )
                axs[-1,j].tick_params(labelbottom=True)
                axs[i,j].set_xlim(min(amplitudes), max(amplitudes))
                axs[i,j].set_ylim(min(frequencies), max(frequencies))
            axs[i,0].set_ylabel('Frequency (Hz)', fontsize='medium')
            axs[i,0].tick_params(labelleft=True)

        ax = fig.add_subplot(spec[:,-1])
        ax.set_frame_on(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Muscle activity (%)", labelpad=10)
        cbar.outline.set_linewidth(0)

        ax = fig.add_subplot(spec[:,0])
        plot_electrode_activation(ax,stim_features['Cathodes'].iloc[0], stim_features['Anodes'].iloc[0])

        plt.tight_layout()
        plt.subplots_adjust(top=0.92,bottom=0.1,left=0.065,right=0.935,hspace=0.24,wspace=0.27)
        name = f"cath{'_'.join(map(str,stim_features['Cathodes'].iloc[0]))}_an{'_'.join(map(str,stim_features['Anodes'].iloc[0]))}"
        plt.suptitle(f"{name}", fontsize='x-large')
        if other_path is None:
            plt.savefig(PATH + f'EMGs_images/Heatmaps/white/{name}.png', transparent=True)
        else:
            plt.savefig(other_path + f'/{name}.png', transparent=True)
        plt.close(fig)


def plot_amplitude_modulation(stim_features, emg_array, amplitude, frequencies, norm='mean', other_path=None):

    actual_frequencies = np.unique(stim_features['Frequency'])
    muscle_perc = 100*np.nanmean(np.abs(emg_array), axis=1) if norm=='mean' else 100*np.nanmax(np.abs(emg_array), axis=1)

    fig, axs = plt.subplots(1,2,figsize=(20,10), gridspec_kw={'width_ratios': [1, 3]})

    axs[1].set_frame_on(False)
    for i_muscle, muscle in enumerate(data.MUSCLES):
        axs[1].scatter(actual_frequencies,muscle_perc[:,i_muscle], label=muscle , marker='x', color=CMAP_MODULATION[i_muscle])#np.max(muscle_perc))
        if not np.any(np.isnan(muscle_perc[:,i_muscle])):
            linear_regression = linear_model.LinearRegression()
            linear_regression.fit(actual_frequencies.reshape(-1,1), muscle_perc[:,i_muscle])
            interpolation = linear_regression.predict(actual_frequencies.reshape(-1,1))
            axs[1].plot(actual_frequencies, interpolation, linewidth=2, color=CMAP_MODULATION[i_muscle])

    axs[1].set_xlabel('Frequency (Hz)', fontsize='20')
    axs[1].set_ylabel("Muscle activity (%)", fontsize='20')
    axs[1].tick_params('both',labelsize = 'xx-large')

    axs[1].set_xlim(min(frequencies), max(frequencies))
    axs[1].set_ylim(0, 100)

    axs[1].legend(ncol=2, fontsize='xx-large', frameon=False, facecolor=(1,1,1,1)) # ncol=2

    plot_electrode_activation(axs[0],stim_features['Cathodes'].iloc[0], stim_features['Anodes'].iloc[0])

    plt.tight_layout()
    plt.subplots_adjust(top=0.91,bottom=0.095,left=0,right=0.965,hspace=0.24,wspace=0)
    name = f"cath{'_'.join(map(str,stim_features['Cathodes'].iloc[0]))}_an{'_'.join(map(str,stim_features['Anodes'].iloc[0]))}_amp{amplitude}"
    plt.suptitle(f"{name} | {amplitude} mA", fontsize='25')
    if other_path is None:
        plt.savefig(PATH + f'EMGs_images/Modulation/{name}.png', transparent=True)
    else:
        plt.savefig(other_path + f'/{name}.png', transparent=True)
    plt.close(fig)


def plot_emgs(stim_features, emg_array, frequencies, amplitude, other_path = None):

    time = np.linspace(0, emg_array.shape[1]/data.FS*1000, emg_array.shape[1])
    fig = plt.figure(figsize=(50,10))
    spec = gridspec.GridSpec(len(data.MUSCLES)//2,len(frequencies)+1)
    axs=[]
    for i in range(len(data.MUSCLES)//2):
        raw = []
        raw.append(fig.add_subplot(spec[i, 1]))
        for j in range(2,len(frequencies)+1):
            raw.append(fig.add_subplot(spec[i, j], sharex=raw[0], sharey=raw[0]))
        axs.append(raw)
    axs=np.array(axs)

    for i_muscle, muscle in enumerate(data.MUSCLES[:len(data.MUSCLES)//2]):
        for i_frequency, frequency in enumerate(frequencies):
            # all frequencies are not founded for one conf and one amplitude
            # need to correctly find the ax corresponding to the actual frequency
            try:
                actual_freq = stim_features["Frequency"].iloc[i_frequency]
                i_actual_freq = np.where(frequencies==actual_freq)[0][0]
                axs[i_muscle, i_actual_freq].plot(time, emg_array[i_frequency,:,i_muscle], CMAP_EMGS[i_muscle], linewidth=5)
                if muscle[:1] != 'TA':
                    axs[i_muscle, i_actual_freq].plot(time, emg_array[i_frequency, :, len(data.MUSCLES)//2+i_muscle], CMAP_EMGS[len(data.MUSCLES)//2 + i_muscle], linewidth=5)
            except:
                None
            axs[i_muscle, i_frequency].tick_params('both', labelbottom=False, labelleft=False)
            axs[i_muscle, i_frequency].ticklabel_format(axis='both', style='sci')
            axs[i_muscle, i_frequency].set_ylim(-1,1)
            axs[-1,i_frequency].tick_params('x', labelbottom=True)
            axs[-1, i_frequency].set_xlabel(f"{frequency} Hz", fontsize='xx-large')
        axs[i_muscle, 0].set_ylabel(f"{muscle[1:]} (mV)", fontsize='xx-large')
        axs[i_muscle,0].tick_params('y', labelleft=True)
        plt.title('time (ms)', fontsize='xx-large', loc='right', y=0)

    ax = fig.add_subplot(spec[:,0])
    plot_electrode_activation(ax,stim_features['Cathodes'].iloc[0], stim_features['Anodes'].iloc[0])

    plt.tight_layout()
    plt.subplots_adjust(left=0.015, bottom=0.075, right=0.985, top=0.940, wspace=0, hspace=0)
    name = f"cath{'_'.join(map(str,stim_features['Cathodes'].iloc[0]))}_an{'_'.join(map(str,stim_features['Anodes'].iloc[0]))}_amp{amplitude}"
    plt.suptitle(f"{name} | {amplitude} mA", fontsize='25')
    if other_path is None:
        plt.savefig(PATH + f'EMGs_images/EMGs/{name}.png', transparent=True, edgecolor='white')
    else:
        plt.savefig(other_path + f'/{name}.png', transparent=True)
    plt.close(fig)
