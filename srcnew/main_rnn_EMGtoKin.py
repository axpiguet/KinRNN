import argparse
import numpy as np
import pandas as pd
import _pickle as cPickle
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy import signal

import data
import rnn
import utils as utils
import tests.params_files
from rnn import BATCH_SIZE, GPU, LR, WORTH_MP
from data import emg

# passing arguments
parser = argparse.ArgumentParser(description='Train an RNN on subject data')
parser.add_argument('ID', metavar='N', type=str, help='ID of the test')
args = vars(parser.parse_args())

ID = args['ID']
from importlib import import_module, __import__
constants = import_module(f'tests.params_files.constants_{ID}')



# load EMG and stimulation data
# old
#DATA = cPickle.load(open(data.PATH + f"/data/{data.SUBJECT_ID}_data_1and2{'_augmented' if constants.SYMMETRY else ''}{'_devmax' if constants.PRE_NORMALIZATION=='mean' else ''}.pkl", "rb" ))


#new
#DATA2 = cPickle.load(open("emg4.pkl", "rb" ),encoding="bytes")
#emg_array, stim_features = data.load(constants.DATA_FEATURES, DATA2, ['LAdd', 'LRF', 'LVLat', 'LST','LTA', 'LMG', 'LSol', 'RAdd', 'RRF', 'RVLat', 'RST', 'RMG', 'RSol'] , fs=data.FS)

#stim_features = stim_features.iloc[10:11,:]


# clean emg data


  #femg_array = emg.filter(emg_array, fs=stable_constants.FS, lowcut=10, highcut=150)

# remove some transcient respose

# PUT IT BACK
#femg_array = femg_array[:,50:,:]

# add nul configuration
#femg_array = np.insert(femg_array,0,0,axis=0)
#stim_features = pd.concat([pd.DataFrame([[20,0,data.PULSE_WIDTH,1,[0],[16],-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]], columns=stim_features.columns),stim_features], ignore_index=True, axis=0)
#stim_features = pd.concat([pd.DataFrame([[20,3,data.PULSE_WIDTH,1,[10],[9],-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]], columns=stim_features.columns),stim_features], ignore_index=True, axis=0)
#stim_features = pd.concat([pd.DataFrame([[20,3.5,data.PULSE_WIDTH,1,[4],[3,5],-1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1]], columns=stim_features.columns),stim_features], ignore_index=True, axis=0)
#stim_features = pd.concat([pd.DataFrame([[20,3.5,data.PULSE_WIDTH,1,[9],[8,10],-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]], columns=stim_features.columns),stim_features], ignore_index=True, axis=0)
# create stim time series
#stim_duration = femg_array.shape[1]*1000/data.FS
#stim_arrays = data.create(stim_features, stim_duration, fs=data.FS)
# tryna plot
#fig, ax = plt.subplots()
#utils.plot_electrode_activation(ax, [10] , [9])
#plt.savefig("graph.png")

#femg_array = data.realign(femg_array, stim_arrays)

#stim_features = stim_features.iloc[0:1,:]
#stim_arrays = stim_arrays[0:1,:,:,:]
#femg_array = femg_array[0:1,:,:]

#stim_features = pd.concat([stim_features, stim_features, stim_features, stim_features, stim_features])
#stim_arrays = np.concatenate((stim_arrays, stim_arrays, stim_arrays, stim_arrays, stim_arrays), axis=0)
#femg_array = np.concatenate((femg_array, femg_array, femg_array, femg_array, femg_array), axis=0)



# split between train and test
#train_stim_features, test_stim_features, train_stim_arrays, test_stim_arrays, train_femg_array, test_femg_array = data.train_test_split(stim_features, stim_arrays, femg_array, test_size=rnn.TEST_SIZE)

#train_stim_features = stim_features
#test_stim_features = stim_features
#train_stim_arrays = stim_arrays
#test_stim_arrays= stim_arrays
#train_femg_array = femg_array
#test_femg_array = femg_array
# augment data

# normalize the training and testing set
#(train_stim_arrays, train_stim_norm), (train_femg_array, train_femg_norm) = data.normalize(train_stim_arrays, per_feature=False), data.normalize(train_femg_array,per_feature=False)
#(test_stim_arrays, _), (test_femg_array,_) = data.normalize(test_stim_arrays, norm=train_stim_norm,per_feature=False), data.normalize(test_femg_array, norm=train_femg_norm,per_feature=False)


# find the configurations and make directories
#reverse_configs, stim_names = utils.make_directories(ID, train_stim_features, constants.PER_CONFIG if type(constants.PER_CONFIG)==bool else True)

# create and save the training sets and targets
#training_sets, training_targets = torch.from_numpy(train_stim_arrays.astype('float32')), torch.nan_to_num(torch.from_numpy(train_femg_array) )  # np.repeat(stim_array, DATASET_SIZE//stim_array.shape[1], axis=1) # np.repeat(femg_array, DATASET_SIZE//(femg_array.shape[1], axis=1)
#testing_sets, testing_targets = torch.from_numpy(test_stim_arrays.astype('float32')), torch.nan_to_num(torch.from_numpy(test_femg_array))

##


#torch.save((training_sets, training_targets), f'{data.PATH}/{ID}/train_sets_targets.pt')
#torch.save((testing_sets, testing_targets), f'{data.PATH}/{ID}/test_sets_targets.pt')
#torch.save((test_stim_features), f'{data.PATH}/{ID}/test_stim_features.pt')

with open('emg54.npy', 'rb') as f:
    helloworld = np.load(f, allow_pickle=True)#[280:580]
    #helloworld = torch.from_numpy(signal.decimate(helloworld[150:750,0:7].T, 3).T.astype('float32')).unsqueeze(dim= 0)
    helloworld = torch.from_numpy(signal.decimate(helloworld[:,0:7].T, 3).T.astype('float32')).unsqueeze(dim= 0)
with open('leftleg.npy', 'rb') as f:
    training_targets = np.load(f, allow_pickle=True)#[280:580]
    #training_targets =0.1*training_targets[150:750]
    #training_targets =0.1*training_targets
    #a = 0.1*training_targets
    training_targets =training_targets
    a =training_targets
    training_targets = torch.from_numpy(signal.decimate(training_targets.T, 3).T.astype('float32')).unsqueeze(dim= 0)
    testing_targets = training_targets
    #testing_targets = torch.from_numpy(testing_targets.astype('float32'))
with open('stimleg.npy', 'rb') as f:
    training_sets = np.load(f, allow_pickle=True)
    #training_sets = training_sets[150:750]
    training_sets = training_sets
    c = training_sets
    training_sets = torch.from_numpy(signal.decimate(training_sets.T, 3,7, ftype='fir').T.astype('float32')).unsqueeze(dim= 0)
    testing_sets = training_sets
    #testing_sets = torch.from_numpy(testing_sets.astype('float32'))

#################### BIGGER DATASET #################################
fileemg = 'emgs40pluss.npy'#'emgs.npy'
with open(fileemg, 'rb') as f:
    helloworld = np.load(f, allow_pickle=True)#[280:580]
    #helloworld = torch.from_numpy(signal.decimate(helloworld[150:750,0:7].T, 3).T.astype('float32')).unsqueeze(dim= 0)
    helloworld = torch.from_numpy(signal.decimate(helloworld[:,:,0:7], 3, axis = 1).astype('float32'))
    print(np.shape(helloworld))
filekin = 'leftlegs40pluss.npy'#'leftlegs.npy'
with open(filekin, 'rb') as f:
    training_targets = np.load(f, allow_pickle=True)#[280:580]
    #training_targets =0.1*training_targets[150:750]
    #training_targets =0.1*training_targets
    #a = 0.1*training_targets
    training_targets =training_targets
    a = training_targets
    training_targets = torch.from_numpy(signal.decimate(training_targets, 3, axis = 1).astype('float32'))
    print('check legs dim ', np.shape(training_targets))
    testing_targets = training_targets
    #testing_targets = torch.from_numpy(testing_targets.astype('float32'))
filestim = 'stimlegs40pluss.npy'#'stimlegs.npy'
with open(filestim, 'rb') as f:
    training_sets =  np.load(f, allow_pickle=True)
    #training_sets = training_sets[150:750]
    #training_sets = torch.from_numpy(signal.decimate(training_sets, 3,7, ftype='fir', axis = 1).astype('float32'))
    testing_sets = training_sets
    #testing_sets = torch.from_numpy(testing_sets.astype('float32'))

dt = 3/1481.48

testing_targets= torch.cat((testing_targets,helloworld), dim =  2)# anlges then emg
training_targets = testing_targets

#torch.save((training_sets, training_targets), f'{data.PATH}/{ID}/train_sets_targets.pt')
#torch.save((testing_sets, testing_targets), f'{data.PATH}/{ID}/test_sets_targets.pt')

from scipy import signal
from scipy.signal import hilbert
b = np.linspace(1, training_sets.shape[1], training_sets.shape[1])#np.linspace(1, 1752., 1752)


#reformer stim:
from scipy.signal import find_peaks
signal_ = np.zeros(np.shape(training_sets))
for k in range(np.shape(training_sets)[0]):
    for elec in range(17):
        peaks, properties = find_peaks(training_sets[k,:,elec], height=0.1)
        print('PEAKS ' , peaks)
        for ind in range(len(peaks)):
            #signal_[0,peaks[ind]-30 : peaks[ind]+30, elec] = properties['peak_heights'][ind]*signal.windows.triang(60)
            signal_[k,peaks[ind]-30 : peaks[ind]+30, elec] = properties['peak_heights'][ind]*signal.windows.triang(60)
        peaks, properties = find_peaks(-training_sets[k,:,elec], height=0.1)
        for ind in range(len(peaks)):
            #signal_[0,peaks[ind]-30 : peaks[ind]+30, elec] = -properties['peak_heights'][ind]*signal.windows.triang(60)
            signal_[k,peaks[ind]-30 : peaks[ind]+30, elec] = -properties['peak_heights'][ind]*signal.windows.triang(60)
        #signal_[0,:,elec] = 20*(np.cos(np.linspace(-9,42,200))+1)
training_sets = signal_.astype('float32') #torch.from_numpy(signal_.astype('float32'))
fig, ax = plt.subplots(1,1)
for i in range (14):
    for z in range (4):
        plt.plot(np.linspace(1, training_sets.shape[1], training_sets.shape[1]), training_sets[z,:,i]-4, 'b')

training_sets = torch.from_numpy(signal.decimate(training_sets, 3,7, ftype='fir', axis = 1).astype('float32'))
#training_sets = torch.from_numpy(training_sets.astype('float32'))
testing_sets = training_sets

for j in range (np.shape(training_sets)[0]):
    training_sets[j]= torch.from_numpy(emg.rolling( training_sets[j,:,:].numpy(),110)[0].astype('float32'))

testing_sets = training_sets

for i in range (14):
    for z in range (4):
        plt.plot(np.linspace(1, 1752, training_sets.shape[1]), training_targets[z,:,2:], 'm')
        plt.plot(np.linspace(1, 1752, training_sets.shape[1]), testing_sets[z,:,i], 'g', linewidth = 3 , alpha = 0.6)
plt.savefig('checkinginput.png')

print('check stim dim ', np.shape(training_sets))




#endregion
#torch.save((training_sets, training_targets), f'{data.PATH}/{ID}/train_sets_targets.pt')
#torch.save((testing_sets, testing_targets), f'{data.PATH}/{ID}/test_sets_targets.pt')


#index_wanted = torch.tensor([1,2,3,4])#[0, 3 , 6, 7])
#training_sets  = torch.index_select(training_sets, 0, index_wanted)
#training_targets  = torch.index_select(training_targets, 0, index_wanted)
#testing_sets = torch.index_select(testing_sets, 0, index_wanted)
#testing_targets  = torch.index_select(testing_targets, 0,index_wanted)

torch.save((training_sets, training_targets), f'{data.PATH}/{ID}/train_sets_targets.pt')
torch.save((testing_sets, testing_targets), f'{data.PATH}/{ID}/test_sets_targets.pt')


#fig, ax = plt.subplots(1, 1)
#plt.plot(np.linspace(0,1000,testing_sets.shape[1]), testing_sets[0], 'b')
#plt.plot(np.linspace(0,1000,testing_sets.shape[1]),emg.rolling(testing_sets[0,:,:].numpy(),135), 'r')
#testing_sets[0]= torch.from_numpy(emg.rolling(testing_sets[0,:,:].numpy(),135)[0].astype('float32'))
#training_sets = testing_sets
#plt.plot(np.linspace(0,1000,testing_sets.shape[1]), testing_sets[0], 'y', alpha = 0.4, linewidth = 5)
#plt.savefig("rolling.png")

#training_sets = torch.tile(testing_sets, (4, 1,1))
#testing_sets = training_sets
#training_targets = torch.tile(testing_targets, (4, 1,1))
#testing_targets = training_targets

#torch.save((training_sets, training_targets), f'{data.PATH}/{ID}/train_sets_targets.pt')
#torch.save((testing_sets, testing_targets), f'{data.PATH}/{ID}/test_sets_targets.pt')


model = eval(f'rnn.{constants.NET}')
net = model(data.N_ELECTRODES, constants.HIDDEN_SIZE, 7 , dt =3* dt)
dev_ = GPU if torch.cuda.is_available() else "cpu"
device_ = torch.device(dev_)
net.to(device_)
#endregion


#region training
#, stim_names
rnn.train_emg(net, ID, n_iterations=constants.N_ITERATIONS, BATCHS =constants.BATCH_SIZE,beta1=constants.BETA1, beta2=constants.BETA2, beta_FR=constants.BETA_FR, config='main', perc_reduction=constants.PERC_REDUCTION, try_checkpoint=True)

#endregion

#torch.save((net.get_alphas()), f'{data.PATH}/{ID}/alphas.pt')

#region testing
print("\nTesting the network...")

# load the net state
loaded_checkpoint = rnn.load_checkpoint('main', ID)
train_loss = loaded_checkpoint['training_losses'][-1]
#test_loss = loaded_checkpoint['testing_losses'][-1]
step = loaded_checkpoint['input_step']
net.load_state_dict(loaded_checkpoint['model_state'])
net.to(torch.device(rnn.GPU if torch.cuda.is_available() else "cpu"))
##
#testing_sets[:,step,:,:]
##
#reverse_configs, input_names = utils.get_configs(test_stim_features)
pred, activity = net(torch.fliplr(torch.rot90(testing_sets, k=-1)).to(device_, non_blocking=True))
fig, ax = plt.subplots(1, 1)
plt.plot(np.linspace(0,1000,testing_targets.shape[1]), testing_targets.cpu()[0,:,0],'x', label = 'true Hip' , color = 'red')
plt.plot(np.linspace(0,1000,testing_targets.shape[1]), testing_targets.cpu()[0,:,1],'x', label = 'true Knee',  color = 'green')
plt.plot(np.linspace(0,1000,testing_targets.shape[1]), pred.detach().cpu()[:,0,0],label = 'pred Hip', color = 'red')
plt.plot(np.linspace(0,1000,testing_targets.shape[1]) , pred.detach().cpu()[:,0,1],label = 'pred Knee',  color = 'green')
plt.legend()
#rnn.plot(net, testing_sets[:,step,:,:], testing_targets, input_names, reverse_configs, rnn.GPU if torch.cuda.is_available() else "cpu", ID, split='Test')
plt.savefig("kinkin.png")
#print(pred[:,0,1])


fig, ax = plt.subplots(1, 1)
for i in range(6):
    plt.plot(np.linspace(0,1000,activity.shape[0]) , activity[:, 0,i].detach().cpu()+0.2*i, linewidth=1)
#ax[1].plot(np.linspace(0,1000,activity.shape[0]) , torch.sigmoid(activity.cpu()[:, 0,:]), linewidth=1)
#ax[1].legend(['0','1','2','3','4','5'])
plt.savefig("acti.png")
print(f"\n\nTrain loss: {train_loss:0.4f} ")#| Test loss: {test_loss:0.4f}\n\n")
#endregion
