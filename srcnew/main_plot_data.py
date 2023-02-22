import _pickle as cPickle
import numpy as np

import data



augment = True
NORM = 'mean'
DATA = cPickle.load(open(data.PATH + f"/data/{data.SUBJECT_ID}_data_1and2{'_augmented'}{'_devmax' if NORM=='mean' else ''}.pkl", "rb" ))
#PATH = f"/media/marion/PARTAGE/Documents/NeuroRestore/Data_backup/mai2021/mai2021_{NORM}norm/{'Augmented/' if augment else 'Non-augmented/'}"
PATH = "C:/Users/yes/Documents/GitHub/little_RNN/"
_, stim_features = data.load({'Session' : ["'mai2021'"], 'Pulses' : [1]}, DATA, data.MUSCLES, fs=data.FS)

unique_electrode_conf = np.unique(stim_features.drop(['Cathodes', 'Anodes', 'Frequency', 'Amplitude', 'PulseWidth', 'Pulses'], axis=1), axis=0)

frequencies = np.unique(stim_features["Frequency"])
n_freq = len(frequencies)
amplitudes = np.unique(stim_features['Amplitude'])

for index in range(len(unique_electrode_conf)):
    dict = {'Session' : ["'mai2021'"],
            'Frequency':frequencies,
            'Pulses':[1],
            'Amplitude': amplitudes}

    for electrode in range(data.N_ELECTRODES):
        dict[f'ElectrodeConf_{electrode+1}'] = [unique_electrode_conf[index,electrode]]

    emg_array, stim_features = data.load(dict, DATA, data.MUSCLES)

    if emg_array is not None:
        emg_array = data.filter(emg_array, fs=data.FS, lowcut=19, highcut=100, order=2)
        emg_array = emg_array[:,50:,:]
        emg_array*=23.247559
        data.plot_bars(stim_features, emg_array, frequencies, amplitudes, norm=NORM, other_path=PATH + f'EMGs_images/Bars')
        data.plot_colormesh(stim_features, emg_array, amplitudes, frequencies, norm=NORM, other_path=PATH + f'EMGs_images/Heatmaps')

    for frequency in frequencies:
        dict['Frequency'] = [frequency]

        emg_array, stim_features = data.load(dict, DATA, data.MUSCLES)

        if emg_array is not None and len(stim_features['Amplitude']) > 1:
            emg_array = data.filter(emg_array, fs=data.FS, lowcut=19, highcut=100, order=2)
            emg_array = emg_array[:,50:,:]
            emg_array*=23.247559
            data.plot_amp_stimvsresponse(stim_features, emg_array, amplitudes, norm=NORM, other_path=PATH + f'EMGs_images/Amplitude_relationship')

    for amplitude in amplitudes:
        dict['Frequency']  = frequencies
        dict['Amplitude'] = [amplitude]

        emg_array, stim_features = data.load(dict, DATA, data.MUSCLES)

        if emg_array is not None :
            emg_array = data.filter(emg_array, fs=data.FS, lowcut=19, highcut=100, order=2)
            emg_array = emg_array[:,50:,:]
            emg_array*=23.247559
            if len(np.unique(stim_features['Frequency'])) > 1 : data.plot_amplitude_modulation(stim_features, emg_array, amplitude, frequencies, norm=NORM, other_path=PATH + f'EMGs_images/Modulation')
            emg_array/=23.247559
            data.plot_emgs(stim_features, emg_array, frequencies, amplitude, other_path=PATH + f'EMGs_images/EMGs')
