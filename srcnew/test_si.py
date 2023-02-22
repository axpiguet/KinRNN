import argparse
import numpy as np
import pandas as pd 
import _pickle as cPickle
import torch

import data
import rnn 
import controller
import utils as utils
import tests.params_files

# passing arguments 
parser = argparse.ArgumentParser(description='Train an RNN on subject data')
parser.add_argument('ID', metavar='N', type=str, help='ID of the test')
args = vars(parser.parse_args())

ID = args['ID']
from importlib import import_module, __import__
constants = import_module(f'tests.params_files.constants_{ID}')


#region data preparation
print("Preparing data...\n")

# load EMG and stimulation data 
DATA = cPickle.load(open(data.PATH + f"/data/{data.SUBJECT_ID}_data_1and2{'_augmented' if constants.SYMMETRY else ''}{'_devmax' if constants.PRE_NORMALIZATION=='mean' else ''}.pkl", "rb" ))
emg_array, stim_features = data.load(constants.DATA_FEATURES, DATA, data.MUSCLES, fs=data.FS)

# clean emg data 
if data.ENVELOPE:
    femg_array = data.filter(emg_array, fs=data.FS, lowcut=79, highcut=101)
else:
  femg_array = data.filter(emg_array, fs=data.FS, lowcut=19, highcut=100, order=2)
  #femg_array = emg.filter(emg_array, fs=stable_constants.FS, lowcut=10, highcut=150)

# remove some transcient respose
femg_array = femg_array[:,50:,:]

# add nul configuration
femg_array = np.insert(femg_array,0,0,axis=0)
stim_features = pd.concat([pd.DataFrame([[20,0,data.PULSE_WIDTH,1,[0],[16],-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]], columns=stim_features.columns),stim_features], ignore_index=True, axis=0) 

# create stim time series
stim_duration = femg_array.shape[1]*1000/data.FS
stim_arrays = data.create(stim_features, stim_duration, fs=data.FS)

femg_array = data.realign(femg_array, stim_arrays)

# split between train and test 
train_stim_features, test_stim_features, train_stim_arrays, test_stim_arrays, train_femg_array, test_femg_array = data.train_test_split(stim_features, stim_arrays, femg_array, test_size=rnn.TEST_SIZE)

# augment data 
train_stim_features['Delay'] = pd.Series(np.full(len(train_stim_features),0), index=train_stim_features.index)
test_stim_features['Delay'] = pd.Series(np.full(len(test_stim_features),0), index=test_stim_features.index)

# normalize the training and testing set
(train_stim_arrays, train_stim_norm), (train_femg_array, train_femg_norm) = data.normalize(train_stim_arrays, per_feature=False), data.normalize(train_femg_array,per_feature=False)
(test_stim_arrays, _), (test_femg_array,_) = data.normalize(test_stim_arrays, norm=train_stim_norm,per_feature=False), data.normalize(test_femg_array, norm=train_femg_norm,per_feature=False)

# create and save the training sets and targets
training_sets, training_targets = torch.from_numpy(train_stim_arrays.astype('float32')), torch.from_numpy(train_femg_array)   # np.repeat(stim_array, DATASET_SIZE//stim_array.shape[1], axis=1) # np.repeat(femg_array, DATASET_SIZE//(femg_array.shape[1], axis=1)
testing_sets, testing_targets = torch.from_numpy(test_stim_arrays.astype('float32')), torch.from_numpy(test_femg_array) 
 
#region load model 
alphas = np.array([rnn.ALPHA.loc[0,freq] for freq in stim_features.loc[:,"Frequency"]])
model = eval(f'rnn.{constants.NET}')
net = model(data.N_ELECTRODES, constants.HIDDEN_SIZE, len(data.MUSCLES), alpha=alphas)

loaded_checkpoint = rnn.load_checkpoint('main', ID)
train_loss = loaded_checkpoint['training_losses'][-1]
test_loss = loaded_checkpoint['testing_losses'][-1]
step = loaded_checkpoint['input_step']
net.load_state_dict(loaded_checkpoint['model_state'])
net.to(rnn.GPU if torch.cuda.is_available() else "cpu")

reverse_configs, input_names = utils.get_configs(test_stim_features)
#endregion


#region testing 
print("\nComputing SI...")

testing_predictions, _ = rnn.predict(net, testing_sets[:,step,:,:].to(torch.device(rnn.GPU if torch.cuda.is_available() else 'cpu')))
testing_predictions = testing_predictions.cpu().numpy()

si_df = test_stim_features.loc[:,['Cathodes', 'Anodes', 'Frequency', 'Amplitude']]
for muscle in data.MUSCLES:
    si_predicted = controller.get_SI(testing_predictions, muscle=muscle, reducted_axis=0)
    si_label = controller.get_SI(testing_targets, muscle=muscle, reducted_axis=1)
    si_error = np.abs(si_label - si_predicted)
    #si_error = np.abs(si_label - si_predicted)/si_label
    si_df[f'Error_SI_{muscle}'] = pd.Series(si_error, index=si_df.index)


import matplotlib.pyplot as plt 
plt.figure()
plt.violinplot(np.array([si_df.loc[:,f'Error_SI_{muscle}'] for muscle in data.MUSCLES]).T, showmeans=True)
plt.xticks(ticks=np.arange(1, len(data.MUSCLES)+1), labels=data.MUSCLES)
plt.ylabel('Selectivity index error')
plt.box(False)
plt.show()
#endregion
