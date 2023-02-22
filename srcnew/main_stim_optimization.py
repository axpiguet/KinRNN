import argparse

import numpy as np 
import pandas as pd
import _pickle as cPickle
import torch 

import data 
import rnn
import controller
import utils

import tests.params_files

from skopt import gp_minimize, gbrt_minimize, forest_minimize
from skopt.space.space import Integer, Real, Categorical

# passing arguments 
parser = argparse.ArgumentParser(description='Train an RNN on subject data')
parser.add_argument('ID', metavar='N', type=str, help='ID of the test')
parser.add_argument('TARGET MUSCLE', metavar='N', type=str, help='muscle to target')
args = vars(parser.parse_args())

ID = args['ID']
TARGET_MUSCLE = args['TARGET MUSCLE']
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
train_stim_features, _, train_stim_arrays, _, train_femg_array, _ = data.train_test_split(stim_features, stim_arrays, femg_array, test_size=rnn.TEST_SIZE)

# augment data 
train_stim_features['Delay'] = pd.Series(np.full(len(train_stim_features),0), index=train_stim_features.index)

# normalize the training and testing set
(train_stim_arrays, train_stim_norm), (train_femg_array, train_femg_norm) = data.normalize(train_stim_arrays, per_feature=False), data.normalize(train_femg_array,per_feature=False)

# create and save the training sets and targets
training_sets, training_targets = torch.from_numpy(train_stim_arrays.astype('float32')), torch.from_numpy(train_femg_array)   # np.repeat(stim_array, DATASET_SIZE//stim_array.shape[1], axis=1) # np.repeat(femg_array, DATASET_SIZE//(femg_array.shape[1], axis=1)
 
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
#endregion

BETA_CATHODE = 4
BETA_ANDODE = 2
BETA_NEUTRAL = 2


#region optimize the stimulation parameters
def stim_optimization(stim_parameters): 
    elec_conf_autoencoder = np.zeros((data.N_ELECTRODES))
    for anode in [stim_parameters[5]]:
        elec_conf_autoencoder[anode] = 1
    for cathode in [stim_parameters[4]]:
        elec_conf_autoencoder[cathode] = -1
    
    stim_parameters = np.array(stim_parameters, dtype=object)
    stim_parameters[5] = [stim_parameters[5]]
    stim_parameters[4] = [stim_parameters[4]]
    stim_parameters = np.concatenate([stim_parameters, elec_conf_autoencoder], dtype=object)
    stim_parameters = np.append(stim_parameters, 0) #offset 

    stim_parameters = pd.DataFrame([stim_parameters], columns=train_stim_features.columns)

    stim_arrays = data.create(stim_parameters, stim_duration, data.FS)
    stim_arrays,_ = data.normalize(stim_arrays, norm=train_stim_norm, per_feature=False)
    stim_tensor = torch.from_numpy(stim_arrays.astype('float32'))

    predicted_femg_array, _ = rnn.predict(net, stim_tensor[:,step,:,:].to(torch.device(rnn.GPU if torch.cuda.is_available() else 'cpu')))
    predicted_femg_array = predicted_femg_array.cpu().numpy()
    si = controller.get_SI(predicted_femg_array, muscle=TARGET_MUSCLE)[0]

    return -si 

print('Optimizing the stimulation...')

dimensions=[Categorical([20,40,80]), 
            Real(0, 5), 
            Categorical([300]), 
            Categorical([1]),# Integer(1,3), 
            Integer(0,15), 
            Integer(0,16)]

res = gbrt_minimize(stim_optimization, dimensions, n_jobs=-1)
opt_input = res.x 
elec_conf_autoencoder = np.zeros((data.N_ELECTRODES))
for anode in [opt_input[5]]:
    elec_conf_autoencoder[anode] = 1
for cathode in [opt_input[4]]:
    elec_conf_autoencoder[cathode] = -1

opt_input = np.array(opt_input, dtype=object)
opt_input[5] = [opt_input[5]]
opt_input[4] = [opt_input[4]]
opt_input = np.concatenate([opt_input, elec_conf_autoencoder], dtype=object)
opt_input = np.append(opt_input, 0) #offset

opt_input = pd.DataFrame([opt_input], columns=train_stim_features.columns)
print(opt_input)
opt_input['Pulses'] = opt_input['Pulses'].astype(int)

opt_input_arrays = data.create(opt_input, stim_duration, data.FS)
opt_input_arrays, _ = data.normalize(opt_input_arrays, norm=train_stim_norm, per_feature=False)
opt_input_tensor = torch.from_numpy(opt_input_arrays.astype('float32'))
opt_pred, _ = rnn.predict(net, opt_input_tensor[:,step,:,:].to(torch.device(rnn.GPU if torch.cuda.is_available() else 'cpu'))) 

cathodes = opt_input['Cathodes'].iloc[0]
anodes = opt_input['Anodes'].iloc[0]

input_name = {"cathodes": cathodes, "anodes": anodes, "name" :  f"cath{'_'.join(map(str,cathodes))}_an{'_'.join(map(str,anodes))}_freq{opt_input['Frequency'].iloc[0]}_amp{opt_input['Amplitude'].iloc[0]}"}

#plot the visual result of the stim in terms of muscular response (modify the code to get to have many cathodes)
controller.plot_result(opt_input_arrays[:,step,:,:], opt_pred.cpu().numpy(), input_name, ID)

#plot the way toward the minimum 
controller.plot_minimization(res.func_vals, TARGET_MUSCLE, ID)

muscle_ind = data.MUSCLES.index(TARGET_MUSCLE)
MaxAbs = controller.get_muscle_response(opt_pred.cpu().numpy(), reducted_axis=0)[0]
print(f'\nFinal {TARGET_MUSCLE} SI = {-res.fun:0.2f} | Muscle response = {MaxAbs[muscle_ind]:0.1f} %')
print(f'\nFinal stim configuration = {input_name["name"]}')
#endregion