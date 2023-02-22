import _pickle as cPickle
import numpy as np 
import pandas as pd

import rnn
import data
import utils 
import argparse
import tests.params_files 
import torch 

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

# test_stim_features = pd.DataFrame([[20,2.5,300,1,0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0],
#                                     [20,6,300,1,0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0],
#                                     [30,2.5,300,1,0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0],
#                                     [20,2.5,300,1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,1,0],

#                                     [30,3.5,300,1,0,0,1,0,0,0,0,0,-1,1,0,0,0,0,0,0,0],

#                                     [30,3.0,300,1,0,0,0,0,0,0,0,0,0,1,-1,0,0,0,0,0,0],
#                                     ], columns=stim_features.columns)
test_stim_features = pd.DataFrame([[20,3.0,300,1,[9],[10]],
                                   [40,3.0,300,1,[9],[10]]
                                    ], columns=stim_features.loc[:,:'Anodes'].columns)
elec_conf_autoencoder = np.zeros((len(test_stim_features), data.N_ELECTRODES))
for index in range(len(test_stim_features)):
  for cathode in test_stim_features['Cathodes'].iloc[index]:
    elec_conf_autoencoder[index, cathode] = -1
  for anode in test_stim_features['Anodes'].iloc[index]:
    elec_conf_autoencoder[index, anode] = 1
elec_conf_autoencoder = pd.DataFrame(elec_conf_autoencoder, columns=[f'ElectrodeConf_{i}' for i in range(1,data.N_ELECTRODES+1)])
test_stim_features = pd.concat([test_stim_features, elec_conf_autoencoder], axis=1)

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
test_stim_arrays = data.create(test_stim_features, stim_duration, fs=data.FS)

femg_array = data.realign(femg_array, stim_arrays)

# split between train and test 
train_stim_features, _, train_stim_arrays, _, train_femg_array, _ = data.train_test_split(stim_features, stim_arrays, femg_array, test_size=rnn.TEST_SIZE)

# augment data 
train_stim_features['Delay'] = pd.Series(np.full(len(train_stim_features),0), index=train_stim_features.index)
test_stim_features['Delay'] = pd.Series(np.full(len(test_stim_features),0), index=test_stim_features.index)

# normalize the training and testing set
(train_stim_arrays, train_stim_norm), (train_femg_array, train_femg_norm) = data.normalize(train_stim_arrays, per_feature=False), data.normalize(train_femg_array,per_feature=False)
test_stim_arrays, _= data.normalize(test_stim_arrays, norm=train_stim_norm,per_feature=False)


# find the configurations and make directories
train_reverse_configs, train_stim_names = utils.get_configs(train_stim_features)

# create and save the training sets and targets
training_sets, training_targets = torch.from_numpy(train_stim_arrays.astype('float32')), torch.from_numpy(train_femg_array)   # np.repeat(stim_array, DATASET_SIZE//stim_array.shape[1], axis=1) # np.repeat(femg_array, DATASET_SIZE//(femg_array.shape[1], axis=1)
testing_sets = torch.from_numpy(test_stim_arrays.astype('float32'))



#region create the network 
alphas = np.array([rnn.ALPHA.loc[0,freq] for freq in stim_features.loc[:,"Frequency"]])
model = eval(f'rnn.{constants.NET}')
net = model(data.N_ELECTRODES, constants.HIDDEN_SIZE, len(data.MUSCLES), alpha=alphas)
loaded_checkpoint = rnn.load_checkpoint('main', ID)
train_loss = loaded_checkpoint['training_losses'][-1]
test_loss = loaded_checkpoint['testing_losses'][-1]
step = loaded_checkpoint['input_step']
net.load_state_dict(loaded_checkpoint['model_state'])
net.to(rnn.GPU if torch.cuda.is_available() else "cpu")
#endregion


#region plot tests
test_reverse_configs, test_input_names = utils.get_configs(test_stim_features)

predicted_emg_array,_ = rnn.predict(net, testing_sets[:,step,:,:].to(torch.device(rnn.GPU if torch.cuda.is_available() else 'cpu')))
predicted_emg_array = predicted_emg_array.cpu().numpy()

inputs_per_config, labels_per_config, pred_per_config = [], [], []
for config in range(len(test_input_names)):
    inputs_per_config.append(test_stim_arrays[:,step,:,:][np.asarray(config==test_reverse_configs),:,:]) 
    pred_per_config.append(predicted_emg_array[:,np.asarray(config==test_reverse_configs),:]) 
 
rnn.plot_pred(inputs_per_config, None, pred_per_config, test_input_names, ID, training_config='main', other_path=data.PATH + '/', split=None)   
#endregion