import argparse
import numpy as np
import pandas as pd
import _pickle as cPickle
import torch
import matplotlib.pyplot as plt

import data
import rnn
import utils as utils
import tests.params_files
from rnn import BATCH_SIZE, GPU, LR, WORTH_MP

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
# old
DATA = cPickle.load(open(data.PATH + f"/data/{data.SUBJECT_ID}_data_1and2.pkl", "rb" ))
emg_array, stim_features = data.load(constants.DATA_FEATURES, DATA, data.MUSCLES, fs=data.FS)

print(DATA[DATA.Pulses == 2][DATA.Session == "march2022"].Frequency.unique())
#new
#DATA2 = cPickle.load(open("emg4.pkl", "rb" ),encoding="bytes")
#emg_array, stim_features = data.load(constants.DATA_FEATURES, DATA2, ['LAdd', 'LRF', 'LVLat', 'LST','LTA', 'LMG', 'LSol', 'RAdd', 'RRF', 'RVLat', 'RST', 'RMG', 'RSol'] , fs=data.FS)

#stim_features = stim_features.iloc[10:11,:]
#emg_array = emg_array[10:11,:,:]
print('Loading correctly performed')

# clean emg data
if data.ENVELOPE:
    femg_array = data.filter(emg_array, fs=data.FS, lowcut=79, highcut=101)
else:
    #femg_array = data.filter(emg_array, fs=data.FS, lowcut=19, highcut=100, order=2)
    femg_array = data.filter(emg_array, fs=data.FS, lowcut=19, highcut=140, order=2)
    print('wassup')
  #femg_array = emg.filter(emg_array, fs=stable_constants.FS, lowcut=10, highcut=150)

# remove some transcient respose

# PUT IT BACK
femg_array = femg_array[:,50:,:]

# add nul configuration
#femg_array = np.insert(femg_array,0,0,axis=0)
#stim_features = pd.concat([pd.DataFrame([[20,0,data.PULSE_WIDTH,1,[0],[16],-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]], columns=stim_features.columns),stim_features], ignore_index=True, axis=0)
#stim_features = pd.concat([pd.DataFrame([[20,3,data.PULSE_WIDTH,1,[10],[9],-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]], columns=stim_features.columns),stim_features], ignore_index=True, axis=0)
#stim_features = pd.concat([pd.DataFrame([[20,3.5,data.PULSE_WIDTH,1,[4],[3,5],-1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1]], columns=stim_features.columns),stim_features], ignore_index=True, axis=0)
#stim_features = pd.concat([pd.DataFrame([[20,3.5,data.PULSE_WIDTH,1,[9],[8,10],-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]], columns=stim_features.columns),stim_features], ignore_index=True, axis=0)
# create stim time series
stim_duration = femg_array.shape[1]*1000/data.FS
stim_arrays = data.create(stim_features, stim_duration, fs=data.FS)
# tryna plot
#fig, ax = plt.subplots()
#utils.plot_electrode_activation(ax, [10] , [9])
#plt.savefig("graph.png")

femg_array = data.realign(femg_array, stim_arrays)

#stim_features = stim_features.iloc[0:1,:]
#stim_arrays = stim_arrays[0:1,:,:,:]
#femg_array = femg_array[0:1,:,:]

#stim_features = pd.concat([stim_features, stim_features, stim_features, stim_features, stim_features])
#stim_arrays = np.concatenate((stim_arrays, stim_arrays, stim_arrays, stim_arrays, stim_arrays), axis=0)
#femg_array = np.concatenate((femg_array, femg_array, femg_array, femg_array, femg_array), axis=0)



# split between train and test
train_stim_features, test_stim_features, train_stim_arrays, test_stim_arrays, train_femg_array, test_femg_array = data.train_test_split(stim_features, stim_arrays, femg_array, test_size=rnn.TEST_SIZE)

#train_stim_features = stim_features
#test_stim_features = stim_features
#train_stim_arrays = stim_arrays
#test_stim_arrays= stim_arrays
#train_femg_array = femg_array
#test_femg_array = femg_array
# augment data
if constants.DATA_AUGMENTATION:
  train_femg_array, train_stim_arrays, train_stim_features = data.augment_data(train_femg_array, train_stim_arrays, train_stim_features, root=2)
  test_femg_array, test_stim_arrays, test_stim_features = data.augment_data(test_femg_array, test_stim_arrays, test_stim_features, root=3)
else:
  train_stim_features['Delay'] = pd.Series(np.full(len(train_stim_features),0), index=train_stim_features.index)
  test_stim_features['Delay'] = pd.Series(np.full(len(test_stim_features),0), index=test_stim_features.index)

# normalize the training and testing set
(train_stim_arrays, train_stim_norm), (train_femg_array, train_femg_norm) = data.normalize(train_stim_arrays, per_feature=False), data.normalize(train_femg_array,per_feature=False)
(test_stim_arrays, _), (test_femg_array,_) = data.normalize(test_stim_arrays, norm=train_stim_norm,per_feature=False), data.normalize(test_femg_array, norm=train_femg_norm,per_feature=False)


# find the configurations and make directories
reverse_configs, stim_names = utils.make_directories(ID, train_stim_features, constants.PER_CONFIG if type(constants.PER_CONFIG)==bool else True)

# create and save the training sets and targets
training_sets, training_targets = torch.from_numpy(train_stim_arrays.astype('float32')), torch.nan_to_num(torch.from_numpy(train_femg_array) )  # np.repeat(stim_array, DATASET_SIZE//stim_array.shape[1], axis=1) # np.repeat(femg_array, DATASET_SIZE//(femg_array.shape[1], axis=1)
testing_sets, testing_targets = torch.from_numpy(test_stim_arrays.astype('float32')), torch.nan_to_num(torch.from_numpy(test_femg_array))

####################################33
training_sets = training_sets[0:1]
training_targets = training_targets [0:1]
testing_sets = training_sets
testing_targets = training_targets

with open('leftleg.npy', 'rb') as f: # 1752 2
    training_targets = np.load(f, allow_pickle=True)
    training_targets = 10*torch.from_numpy(training_targets.astype('float32'))[280:580].unsqueeze(dim= 0)
    testing_targets = training_targets
with open('stimleg.npy', 'rb') as f: # 1752 17
    training_sets = np.load(f, allow_pickle=True)
    training_sets = 10*torch.from_numpy(training_sets.astype('float32'))[250:550].unsqueeze(dim= 0).tile((5,1,1)).unsqueeze(dim= 0)
    testing_sets = training_sets
#with open('emg54.npy', 'rb') as f: # 1752 14
#    training_targets = np.load(f, allow_pickle=True)
#    training_targets = torch.from_numpy(training_targets.astype('float32'))[280:580].unsqueeze(dim= 0)
#    testing_targets = training_targets
print('STIM                   ' , np.shape(testing_sets))
print('OUT                 ' , np.shape(testing_targets))
####################################33
torch.save((training_sets, training_targets), f'{data.PATH}/{ID}/train_sets_targets.pt')
torch.save((testing_sets, testing_targets), f'{data.PATH}/{ID}/test_sets_targets.pt')
torch.save((test_stim_features), f'{data.PATH}/{ID}/test_stim_features.pt')

#endregion
print(training_sets.shape, training_targets.shape)
print(testing_sets.shape, testing_targets.shape)
print(np.sum(~np.isnan(testing_targets[0,:,:].numpy())))
#region create model
alphas = np.array([rnn.ALPHA.loc[0,int(freq)] for freq in stim_features.loc[:,"Frequency"]])
print(alphas)
model = eval(f'rnn.{constants.NET}')
#net = model(data.N_ELECTRODES, constants.HIDDEN_SIZE, len(data.MUSCLES), alpha=alphas)
net = model(data.N_ELECTRODES, constants.HIDDEN_SIZE, 2, alpha=alphas)
dev_ = GPU if torch.cuda.is_available() else "cpu"
device_ = torch.device(dev_)
net.to(device_)
#endregion


#region training
if (constants.PER_CONFIG if type(constants.PER_CONFIG)==bool else True):
  rnn.train_per_configs(net, train_stim_arrays, train_femg_array, stim_names, reverse_configs, constants.PER_CONFIG, alphas, ID, n_iterations=constants.N_ITERATIONS, beta1=constants.BETA1, beta2=constants.BETA2, beta_FR=constants.BETA_FR, perc_reduction=constants.PERC_REDUCTION)
  print('loop1')
else :
  rnn.train(net, stim_names, reverse_configs, ID, n_iterations=constants.N_ITERATIONS, beta1=constants.BETA1, beta2=constants.BETA2, beta_FR=constants.BETA_FR, config='main', perc_reduction=constants.PERC_REDUCTION, try_checkpoint=True)
  print('loop2')
#endregion
#pred, activity = net(torch.fliplr(torch.rot90(testing_sets, k=-1)).to(device_, non_blocking=True))

pred, activity = net(torch.fliplr(torch.rot90(testing_sets[:,0,:,:], k=-1)).to(device_, non_blocking=True))
#print(net.get_weightsgrad())
fig, ax = plt.subplots(1, 1)
plt.plot(np.linspace(0,300,testing_targets.shape[1]), testing_targets[0,:,0].cpu().detach().numpy(),'x', label = 'true Hip' , color = 'indianred', alpha = 0.5)
plt.plot(np.linspace(0,300,testing_targets.shape[1]), testing_targets[0,:,1].cpu().detach().numpy(),'x', label = 'true Knee',  color = 'forestgreen', alpha = 0.5)
plt.plot(np.linspace(0,300,testing_targets.shape[1]), pred[:,0,0].cpu().detach().numpy(),label = 'true Hip', color = 'red')
plt.plot(np.linspace(0,300,testing_targets.shape[1]) , pred[:,0,1].cpu().detach().numpy(),label = 'true Knee',  color = 'green')
plt.legend()
plt.savefig("kkoko.png")
#torch.save((net.get_alphas()), f'{data.PATH}/{ID}/alphas.pt')

#region testing
print("\nTesting the network...")

# load the net state
loaded_checkpoint = rnn.load_checkpoint('main', ID)
train_loss = loaded_checkpoint['training_losses'][-1]
test_loss = loaded_checkpoint['testing_losses'][-1]
step = loaded_checkpoint['input_step']
net.load_state_dict(loaded_checkpoint['model_state'])
net.to(torch.device(rnn.GPU if torch.cuda.is_available() else "cpu"))
##
#testing_sets[:,step,:,:]
##
#reverse_configs, input_names = utils.get_configs(test_stim_features)
reverse_configs, input_names = utils.get_configs(test_stim_features[0:1])
print(testing_sets[:,step,:,:].shape, testing_targets.shape)
rnn.plot(net, testing_sets[:,step,:,:], testing_targets, input_names, reverse_configs, rnn.GPU if torch.cuda.is_available() else "cpu", ID, split='Test')
plt.savefig("lstmessai.png")
print(f"\n\nTrain loss: {train_loss:0.4f} | Test loss: {test_loss:0.4f}\n\n")
#endregion
