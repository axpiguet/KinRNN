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

# passing arguments
parser = argparse.ArgumentParser(description='Train an RNN on subject data')
parser.add_argument('ID', metavar='N', type=str, help='ID of the test')
args = vars(parser.parse_args())

ID = args['ID']
from importlib import import_module, __import__
constants = import_module(f'tests.params_files.constants_{ID}')
print(ID)

#region data preparation
print("Preparing data...\n")

# load EMG and stimulation data
#DATA = cPickle.load(open(data.PATH + f"/data/{data.SUBJECT_ID}_data_1and2{'_augmented' if constants.SYMMETRY else ''}{'_devmax' if constants.PRE_NORMALIZATION=='mean' else ''}.pkl", "rb" ))

#emg_array, stim_features = data.load(constants.DATA_FEATURES, DATA, data.MUSCLES, fs=data.FS)

#DATA = cPickle.load(open("emg4.pkl", "rb" ),encoding="bytes")
#emg_array, stim_features = data.load(constants.DATA_FEATURES, DATA, ['LAdd', 'LRF', 'LVLat', 'LST','LTA', 'LMG', 'LSol', 'RAdd', 'RRF', 'RVLat', 'RST', 'RMG', 'RSol'] , fs=data.FS)

DATA2 = cPickle.load(open("emg4.pkl", "rb" ),encoding="bytes")
emg_array, stim_features = data.load(constants.DATA_FEATURES, DATA2, ['LAdd', 'LRF', 'LVLat', 'LST','LTA', 'LMG', 'LSol', 'RAdd', 'RRF', 'RVLat', 'RST', 'RMG', 'RSol'] , fs=data.FS)

#stim_features = stim_features.iloc[10:11,:]
#emg_array = emg_array[10:11,:,:]
#pred1 = torch.load( f'{data.PATH}/{ID}/test_pred.pt')

# clean emg data
if data.ENVELOPE:
    femg_array = data.filter(emg_array, fs=data.FS, lowcut=79, highcut=101)
else:
    #femg_array = data.filter(emg_array, fs=data.FS, lowcut=19, highcut=100, order=2)
    femg_array = data.filter(emg_array, fs=data.FS, lowcut=19, highcut=130, order=2)
  #femg_array = emg.filter(emg_array, fs=stable_constants.FS, lowcut=10, highcut=150)

# remove some transcient respose
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





#region create model
alphas = np.array([rnn.ALPHA.loc[0,freq] for freq in stim_features.loc[:,"Frequency"]])
model = eval(f'rnn.{constants.NET}')
net = model(data.N_ELECTRODES, constants.HIDDEN_SIZE, len(data.MUSCLES), alpha=alphas)
#endregion

#region testing
print("\nTesting the network...")

testing_sets, testing_targets = torch.load( f'{data.PATH}/{ID}/test_sets_targets.pt')
test_stim_features = torch.load( f'{data.PATH}/{ID}/test_stim_features.pt')


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
reverse_configs, input_names = utils.get_configs(test_stim_features)
rnn.plot(net, testing_sets[:,step,:,:], testing_targets, input_names, reverse_configs, rnn.GPU if torch.cuda.is_available() else "cpu", ID, split='Test')
plt.savefig("lstmessai.png")
print(f"\n\nTrain loss: {train_loss:0.4f} | Test loss: {test_loss:0.4f}\n\n")
#endregion

reverse_configs, input_names = utils.get_configs(test_stim_features)
_, ind_configs, reverse_configs = np.unique(test_stim_features.drop(['Frequency','Amplitude', 'Delay','Cathodes','Anodes'], axis=1), axis=0, return_index=True, return_inverse=True)

amps = []
musc = ['LAdd', 'LRF', 'LVLat', 'LST','LTA', 'LMG', 'LSol', 'RAdd', 'RRF', 'RVLat', 'RST', 'RMG', 'RSol']
df_diff = pd.DataFrame(columns = musc)
pred,_ = rnn.predict(net, testing_sets[:,step,:,:].float().to(torch.device(rnn.GPU if torch.cuda.is_available() else 'cpu')))
pred = pred.cpu().numpy()
pred = np.fliplr(np.rot90(pred, k=-1))
for config in range(len(ind_configs)):
    muscle_perc,actual_amplitudes = data.recrutement(test_stim_features.iloc[np.where(np.asarray(config==reverse_configs)==True)[0],:], testing_targets[np.where(np.asarray(config==reverse_configs)==True)[0].tolist(),:,:], frequencies=constants.DATA_FEATURES['Frequency'],norm=constants.PRE_NORMALIZATION)
    muscle_perchat,actual_amplitudeshat = data.recrutement(test_stim_features.iloc[np.where(np.asarray(config==reverse_configs)==True)[0],:], pred[np.where(np.asarray(config==reverse_configs)==True)[0].tolist(),:,:], frequencies=constants.DATA_FEATURES['Frequency'],norm=constants.PRE_NORMALIZATION)
    df_diff = df_diff.append(pd.DataFrame(np.abs(muscle_perc-muscle_perchat), columns=df_diff.columns), ignore_index=True)
    for i in range(muscle_perc.shape[0]):
        amps.append(actual_amplitudes[0])
df_diff['Amp'] = amps
print(df_diff.shape)
print(df_diff.head())
analysis = df_diff.dropna()
grouped = pd.DataFrame(columns = ['Amp'])
grouped['Amp'] = analysis['Amp']
grouped['average'] = analysis[musc].mean(axis=1)
print(grouped.groupby(['Amp'])['average'])
#.agg("mean", axis="columns")
for col in musc :
    print('% error for ',col , ' : ' ,int(analysis[col].mean()), ' %')


torch.save((pred), f'{data.PATH}/{ID}/test_pred.pt')


from sklearn.metrics import r2_score
from rnn.predict import predict

print('R squared score flatten : ' ,r2_score(testing_targets.flatten(), pred.flatten()))
print('R squared score : ' ,r2_score(testing_targets.T.reshape((testing_targets.T.shape[0], -1)), pred.T.reshape((pred.T.shape[0], -1))  ))
print('R squared score : ' ,r2_score(testing_targets.T.reshape((testing_targets.T.shape[0], -1)), testing_targets.T.reshape((testing_targets.T.shape[0], -1))))
