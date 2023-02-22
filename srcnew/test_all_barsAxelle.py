from audioop import reverse
import _pickle as cPickle
import numpy as np
import pandas as pd
import argparse
import torch

import rnn
import data
import utils
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
#if data.ENVELOPE:
#    femg_array = data.filter(emg_array, fs=data.FS, lowcut=79, highcut=101)
#else:
#  femg_array = data.filter(emg_array, fs=data.FS, lowcut=19, highcut=100, order=2)
  #femg_array = emg.filter(emg_array, fs=stable_constants.FS, lowcut=10, highcut=150)

# remove some transcient respose
#femg_array = femg_array[:,50:,:]

# add nul configuration
#femg_array = np.insert(femg_array,0,0,axis=0)
#stim_features = pd.concat([pd.DataFrame([[20,0,data.PULSE_WIDTH,1,[0],[16],-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]], columns=stim_features.columns),stim_features], ignore_index=True, axis=0)

# create stim time series
#stim_duration = femg_array.shape[1]*1000/data.FS
#stim_arrays = data.create(stim_features, stim_duration, fs=data.FS)

#femg_array = data.realign(femg_array, stim_arrays)

# split between train and test
#train_stim_features, test_stim_features, train_stim_arrays, test_stim_arrays, train_femg_array, test_femg_array = data.train_test_split(stim_features, stim_arrays, femg_array, test_size=rnn.TEST_SIZE)

# normalize the training and testing set
#(train_stim_arrays, train_stim_norm), (train_femg_array, train_femg_norm) = data.normalize(train_stim_arrays, per_feature=False), data.normalize(train_femg_array,per_feature=False)
#(test_stim_arrays, _), (test_femg_array, _) = data.normalize(test_stim_arrays, norm=train_stim_norm, per_feature=False), data.normalize(test_femg_array, norm=train_femg_norm, per_feature=False)

#test_stim_features['Delay'] = pd.Series(np.full(len(test_stim_features),0), index=test_stim_features.index)
# create and save the training sets and targets
##test_inputs = torch.from_numpy(test_stim_arrays.astype('float32'))
#endregion
######################################## remove above

test_inputs, test_femg_array = torch.load( f'{data.PATH}/{ID}/test_sets_targets.pt')
test_stim_features = torch.load(  f'{data.PATH}/{ID}/test_stim_features.pt')
stim_features = test_stim_features
#region create the network
alphas = np.array([rnn.ALPHA.loc[0,freq] for freq in stim_features.loc[:,"Frequency"]])
model = eval(f'rnn.{constants.NET}')
net = model(data.N_ELECTRODES, constants.HIDDEN_SIZE, len(data.MUSCLES), alpha=alphas)
loaded_checkpoint = rnn.load_checkpoint('main', ID)
train_loss = loaded_checkpoint['training_losses'][-1]
test_loss = loaded_checkpoint['testing_losses'][-1]
step = loaded_checkpoint['input_step']
net.load_state_dict(loaded_checkpoint['model_state'])
net.to(torch.device(rnn.GPU if torch.cuda.is_available() else "cpu"))
#endregion


predicted_emg_array,_ = rnn.predict(net, test_inputs[:,step,:,:].float().to(torch.device(rnn.GPU if torch.cuda.is_available() else 'cpu')))
predicted_emg_array = predicted_emg_array.cpu().numpy()
predicted_emg_array = np.fliplr(np.rot90(predicted_emg_array, k=-1))

predicted_emg_array/=100
predicted_emg_array*=23.247559 # because we are going to quantify by the mean we need to come back to the mean quantified emg but without devmax

test_femg_array/=100
test_femg_array*=23.247559 # because we are going to quantify by the mean we need to come back to the mean quantified emg but without devmax

amplitudes_range = np.arange(0, 5.5, 0.5)


#region do each plot
_, ind_configs, reverse_configs = np.unique(test_stim_features.drop(['Frequency','Amplitude', 'Delay','Cathodes','Anodes'], axis=1), axis=0, return_index=True, return_inverse=True)

for config in range(len(ind_configs)):
    plot_path =  f'{data.PATH}\{ID}\Bars'
    #data.plot_bars(test_stim_features.iloc[np.asarray(config==reverse_configs),:], predicted_emg_array.iloc[np.asarray(config==reverse_configs),:,:], frequencies=constants.DATA_FEATURES['Frequency'], amplitudes=amplitudes_range ,norm=constants.PRE_NORMALIZATION, other_path=plot_path+'predicted/')
    #data.plot_bars(test_stim_features.iloc[np.asarray(config==reverse_configs),:], test_femg_array.iloc[np.asarray(config==reverse_configs),:,:], frequencies=constants.DATA_FEATURES['Frequency'], amplitudes=amplitudes_range ,norm=constants.PRE_NORMALIZATION, other_path=plot_path+'measured/')
    data.plot_bars(test_stim_features.iloc[np.where(np.asarray(config==reverse_configs)==True)[0],:], predicted_emg_array[np.where(np.asarray(config==reverse_configs)==True)[0].tolist(),:,:], frequencies=constants.DATA_FEATURES['Frequency'], amplitudes=amplitudes_range ,norm=constants.PRE_NORMALIZATION, other_path=plot_path+'\predicted')
    data.plot_bars(test_stim_features.iloc[np.where(np.asarray(config==reverse_configs)==True)[0],:], test_femg_array[np.where(np.asarray(config==reverse_configs)==True)[0],:,:], frequencies=constants.DATA_FEATURES['Frequency'], amplitudes=amplitudes_range ,norm=constants.PRE_NORMALIZATION, other_path=plot_path+'\measured')
