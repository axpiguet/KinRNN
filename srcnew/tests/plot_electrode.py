import _pickle as cPickle 
import data 
import utils
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

DATA = cPickle.load(open(data.PATH + f"/data/{data.SUBJECT_ID}_data_1and2{'_augmented'}{'_devmax'}.pkl", "rb" ))
dict = {'Pulses':[1]}
emg_array, stim_features = data.load(dict, DATA, data.MUSCLES)
stim_features['Delay'] = pd.Series(np.full(len(stim_features),0), index=stim_features.index)

_, stim_names = utils.get_configs(stim_features)
for config in range(len(stim_names)):
    fig = plt.figure(figsize=(5,10))
    ax = fig.add_subplot()
    utils.plot_electrode_activation(ax, stim_names['cathodes'][config], stim_names["anodes"][config])
    name = f"cath{'_'.join(map(str,stim_names['cathodes'][config]))}_an{'_'.join(map(str,stim_names['anodes'][config]))}"
    plt.savefig(f'D:/Documents/NeuroRestore/Presentation_materials/Electrodes/{name}.png', transparent=True)
    plt.close()