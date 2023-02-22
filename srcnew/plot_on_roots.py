import numpy as np
import pandas as pd 
import argparse
import _pickle as cPickle
from sklearn import utils
import torch

import data
import rnn 
import utils

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

# passing arguments 
parser = argparse.ArgumentParser(description='Train an RNN on subject data')
parser.add_argument('ID', metavar='N', type=str, help='ID of the test')
args = vars(parser.parse_args())

ID = args['ID']
BETA_CATHODE = 4
BETA_ANDODE = 2
BETA_NEUTRAL = 2
stim_duration = 250

from importlib import import_module, __import__
constants = import_module(f'tests.params_files.constants_{ID}')


#region data preparation
print("Preparing data...\n")

# load EMG and stimulation data 
DATA = cPickle.load(open(data.PATH + f"/data/{data.SUBJECT_ID}_data_1and2{'_augmented' if constants.SYMMETRY else ''}{'_devmax' if constants.PRE_NORMALIZATION=='mean' else ''}.pkl", "rb" ))
_, stim_features = data.load(constants.DATA_FEATURES, DATA, data.MUSCLES, fs=data.FS)


cathode_stim_array = -np.eye(16)
cathode_stim_features = pd.DataFrame(cathode_stim_array, columns = [f'ElectrodeConf_{i}' for i in range(1,17)])
cathode_stim_features['Frequency'] = 20
cathode_stim_features['Amplitude'] = pd.Series(np.full(len(cathode_stim_features),3.5), index=cathode_stim_features.index)
cathode_stim_features['PulseWidth'] = pd.Series(np.full(len(cathode_stim_features),data.PULSE_WIDTH), index=cathode_stim_features.index)
cathode_stim_features['Pulses'] = pd.Series(np.full(len(cathode_stim_features),1), index=cathode_stim_features.index)
cathode_stim_features['Cathodes'] = pd.Series([[i] for i in range(len(cathode_stim_features))], index=cathode_stim_features.index)
cathode_stim_features['Anodes'] = pd.Series([[16] for i in range(len(cathode_stim_features))], index=cathode_stim_features.index)
cathode_stim_features['ElectrodeConf_17'] = pd.Series(np.full(len(cathode_stim_features),1), index=cathode_stim_features.index)

# add nul configuration
stim_features = pd.concat([pd.DataFrame([[20,0,data.PULSE_WIDTH,1,[0],[16],-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]], columns=stim_features.columns),stim_features], ignore_index=True, axis=0) 

# create stim time series
stim_duration = 250
stim_arrays = data.create(stim_features, stim_duration, fs=data.FS)
cathode_stim_arrays = data.create(cathode_stim_features, stim_duration, fs=data.FS)

# split between train and test 
train_stim_features, _, train_stim_arrays, _, _, _ = data.train_test_split(stim_features, stim_arrays, np.zeros((stim_features.shape[0], stim_duration, len(data.MUSCLES))), test_size=rnn.TEST_SIZE)

# augment data 
train_stim_features['Delay'] = pd.Series(np.full(len(train_stim_features),0), index=train_stim_features.index)
cathode_stim_features['Delay'] = pd.Series(np.full(len(cathode_stim_features),0), index=cathode_stim_features.index)

# normalize the training and testing set
train_stim_arrays, train_stim_norm = data.normalize(train_stim_arrays, per_feature=False)
cathode_stim_arrays, _= data.normalize(cathode_stim_arrays, norm=train_stim_norm,per_feature=False)

# create and save the training sets and targets
training_sets = torch.from_numpy(train_stim_arrays.astype('float32'))  # np.repeat(stim_array, DATASET_SIZE//stim_array.shape[1], axis=1) # np.repeat(femg_array, DATASET_SIZE//(femg_array.shape[1], axis=1)
cathode_sets = torch.from_numpy(cathode_stim_arrays.astype('float32')) 
 
#region load model 
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



#region calculate the barycenters and plot it on roots

# for each cathode and a given cathode predict the activity of the network (at 20Hz, 1 pulse and 3.5mA)
pred, neuronal_activity = rnn.predict(net, cathode_sets[:,step,:,:].to(torch.device(rnn.GPU if torch.cuda.is_available() else 'cpu'))) 
pred, neuronal_activity = pred.cpu().numpy(), neuronal_activity[100:,:,:].cpu().numpy()

# for each neuron, calculate the peak to peak for all of those activity 
peak2peak = np.abs(np.max(neuronal_activity, axis=0)- np.min(neuronal_activity, axis=0))

# determine for each neuron in the output weight matrix the muscle toward which the connexion is the more important 
output_weight = net.get_weights()['W']

ind_max_weight = np.argmax(output_weight.cpu().detach().numpy(), axis=0)
# determine a colormap 

# place all neurons on the graph with barycenter 
sum_cathodes = []
for cathode in range(peak2peak.shape[0]):
    sum_cathodes.append(np.array([peak2peak[cathode,:]*utils.ELECTRODE_POSITIONS_ELEC[cathode][0],peak2peak[cathode,:]*utils.ELECTRODE_POSITIONS_ELEC[cathode][1]]))
sum_cathodes = np.array(sum_cathodes)

barycenters = np.sum(sum_cathodes, axis=0)/np.sum(peak2peak, axis=0) # division element par element 

fig = plt.figure() # 30 15 
spec = gridspec.GridSpec(1,2)

ax = fig.add_subplot(spec[:,0])
utils.plot_root_neurons(ax, barycenters, ind_max_weight)
plt.savefig('hello.png', transparent=True)
#endregion