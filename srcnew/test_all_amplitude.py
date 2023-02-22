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

# normalize the training and testing set
(train_stim_arrays, train_stim_norm), (train_femg_array, train_femg_norm) = data.normalize(train_stim_arrays, per_feature=False), data.normalize(train_femg_array,per_feature=False)
(test_stim_arrays, _), (test_femg_array, _) = data.normalize(test_stim_arrays, norm=train_stim_norm, per_feature=False), data.normalize(test_femg_array, norm=train_femg_norm, per_feature=False)

stim_features = pd.concat([train_stim_features, test_stim_features], axis=0)
stim_arrays = np.concatenate([train_stim_arrays, test_stim_arrays], axis=0)
femg_array = np.concatenate([train_femg_array, test_femg_array], axis=0)

stim_features['Delay'] = pd.Series(np.full(len(stim_features),0), index=stim_features.index)
# create and save the training sets and targets
inputs = torch.from_numpy(stim_arrays.astype('float32'))
#endregion


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


#region do each plot 
amplitudes_range = np.arange(0, 5.5, 0.5)
# find the configurations and make directories
reverse_configs, stim_names = utils.get_configs(stim_features) # one config is just one freq one electrode conf and many amplitudes (we set delay to 1)
inputs_features_per_config, inputs_per_config, labels_per_config = [], [], []
for config in range(len(stim_names)):
  inputs_features_per_config.append(stim_features.iloc[np.asarray(config==reverse_configs),:])
  inputs_per_config.append(inputs[np.asarray(config==reverse_configs),:,:])
  labels_per_config.append(femg_array[np.asarray(config==reverse_configs),:,:])

for config in range(len(stim_names)):
  actual_amplitudes = np.array(inputs_features_per_config[config]['Amplitude'])
  if len(actual_amplitudes) > 2:
    print(actual_amplitudes)
    test_amplitudes = []
    for i_amp, amplitude in enumerate(amplitudes_range):
          if amplitude not in actual_amplitudes:
              test_amplitudes.append(amplitude)

    test_amplitudes = np.concatenate((actual_amplitudes, test_amplitudes))
    arg_sort_amplitudes = np.argsort(test_amplitudes)
    test_amplitudes = np.array(test_amplitudes)[arg_sort_amplitudes]

    test_stim_features = pd.DataFrame(np.array([np.array(test_amplitudes).T]).T, columns=['Amplitude'])
    test_stim_features['Frequency'] = pd.Series(np.full(len(test_stim_features), inputs_features_per_config[config]['Frequency'].iloc[0]), index=test_stim_features.index)
    test_stim_features['PulseWidth'] =  pd.Series(np.full(len(test_stim_features), inputs_features_per_config[config]['PulseWidth'].iloc[0]), index=test_stim_features.index)
    test_stim_features['Pulses'] =  pd.Series(np.full(len(test_stim_features), inputs_features_per_config[config]['Pulses'].iloc[0]), index=test_stim_features.index)
    test_stim_features['Cathodes'] =  pd.Series([inputs_features_per_config[config]['Cathodes'].iloc[0] for i in range(len(test_stim_features))], index=test_stim_features.index)
    test_stim_features['Anodes'] =  pd.Series([inputs_features_per_config[config]['Anodes'].iloc[0] for i in range(len(test_stim_features))], index=test_stim_features.index)
    test_stim_features['Delay'] = pd.Series(np.full(len(test_stim_features),0), index=test_stim_features.index)
    
    elec_conf_autoencoder = np.zeros((len(test_stim_features), data.N_ELECTRODES))
    for index in range(len(test_stim_features)):
      for cathode in test_stim_features['Cathodes'].iloc[index]:
        elec_conf_autoencoder[index, cathode] = -1
      for anode in test_stim_features['Anodes'].iloc[index]:
        elec_conf_autoencoder[index, anode] = 1
    elec_conf_autoencoder = pd.DataFrame(elec_conf_autoencoder, columns=[f'ElectrodeConf_{i}' for i in range(1,data.N_ELECTRODES+1)])
    test_stim_features = pd.concat([test_stim_features, elec_conf_autoencoder], axis=1)

    test_stim_arrays = data.create(test_stim_features, stim_duration, fs=data.FS)
    test_stim_arrays, _ = data.normalize(test_stim_arrays, norm=train_stim_norm, per_feature=False)
    test_inputs = torch.from_numpy(test_stim_arrays.astype('float32'))

    predicted_emg_array,_ = rnn.predict(net, test_inputs[:,step,:,:].float().to(torch.device(rnn.GPU if torch.cuda.is_available() else 'cpu')))
    predicted_emg_array = predicted_emg_array.cpu().numpy()
    predicted_emg_array = np.fliplr(np.rot90(predicted_emg_array, k=-1))
  

    arg_sort_label  = np.argsort(actual_amplitudes)

    label_stim_features = inputs_features_per_config[config].iloc[arg_sort_label,:]
    label_emg_array = labels_per_config[config][arg_sort_label,:,:]

    predicted_emg_array/=100
    predicted_emg_array*=23.247559 # because we are going to quantify by the mean we need to come back to the mean quantified emg but without devmax

    label_emg_array/=100
    label_emg_array*=23.247559 # because we are going to quantify by the mean we need to come back to the mean quantified emg but without devmax

    plot_path =  f'/media/marion/PARTAGE/Documents/NeuroRestore/Results_backup/mai2021/Amplitude_relationship/'
    data.plot_amp_stimvsresponse(test_stim_features, predicted_emg_array, amplitudes_range, label_stim_features=label_stim_features, label_emg_array=label_emg_array, norm=constants.PRE_NORMALIZATION, other_path=plot_path)