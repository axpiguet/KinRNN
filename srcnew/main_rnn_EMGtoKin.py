import argparse
import numpy as np
import pandas as pd
import _pickle as cPickle
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy import signal
from importlib import import_module, __import__
import data
import rnn
import utils as utils
import tests.params_files
from rnn import BATCH_SIZE, GPU, LR, WORTH_MP
from data import emg
from scipy.signal import hilbert
from scipy.signal import find_peaks

# Passing arguments
parser = argparse.ArgumentParser(description="Train an RNN on subject data")
parser.add_argument("ID", metavar="N", type=str, help="ID of the test")
args = vars(parser.parse_args())

ID = args["ID"]
constants = import_module(f"tests.params_files.constants_{ID}")
dt = 3 / 1481.48

#################### Load DATASET #################################
fileemg = "emgs40pluss.npy"
filekin = "leftlegs40pluss.npy"
filestim = "stimlegs40pluss.npy"

with open(fileemg, "rb") as f:
    emg_data = np.load(f, allow_pickle=True)
    emg_data = torch.from_numpy(signal.decimate(emg_data[:, :, 0:7], 3, axis=1).astype("float32"))

with open(filekin, "rb") as f:
    training_targets = np.load(f, allow_pickle=True)
    training_targets = torch.from_numpy(signal.decimate(training_targets, 3, axis=1).astype("float32"))
    testing_targets = training_targets

with open(filestim, "rb") as f:
    training_sets = np.load(f, allow_pickle=True)
    testing_sets = training_sets

# Rearrange
testing_targets = torch.cat((testing_targets, emg_data), dim=2)  # Anlges, then EMG
training_targets = testing_targets

# reformer stim:
signal_ = np.zeros(np.shape(training_sets))
for k in range(np.shape(training_sets)[0]):
    for elec in range(17):
        peaks, properties = find_peaks(training_sets[k, :, elec], height=0.1)
        for ind in range(len(peaks)):
            signal_[k, peaks[ind] - 30 : peaks[ind] + 30, elec] = properties["peak_heights"][ind] * signal.windows.triang(60)
        peaks, properties = find_peaks(-training_sets[k, :, elec], height=0.1)
        for ind in range(len(peaks)):
            signal_[k, peaks[ind] - 30 : peaks[ind] + 30, elec] = -properties["peak_heights"][ind] * signal.windows.triang(60)
training_sets = signal_.astype("float32")

fig, ax = plt.subplots(1, 1)
for i in range(training_sets.shape[2]):
    for z in range(training_sets.shape[0]):
        plt.plot(np.linspace(1, training_sets.shape[1], training_sets.shape[1]),training_sets[z, :, i] - 4,"b")

training_sets = torch.from_numpy(signal.decimate(training_sets, 3, 7, ftype="fir", axis=1).astype("float32"))
testing_sets = training_sets

for j in range(np.shape(training_sets)[0]):
    training_sets[j] = torch.from_numpy(emg.rolling(training_sets[j, :, :].numpy(), 110)[0].astype("float32"))

testing_sets = training_sets

for i in range(training_sets.shape[2]):
    for z in range(training_sets.shape[0]):
        plt.plot(np.linspace(1, 1752, training_sets.shape[1]),training_targets[z, :, 2:],"m")
        plt.plot(
            np.linspace(1, 1752, training_sets.shape[1]),
            testing_sets[z, :, i],
            "g",
            linewidth=3,
            alpha=0.6,
        )
plt.savefig("checkinginput.png")


# Saving datasets for training

torch.save((training_sets, training_targets), f"{data.PATH}/{ID}/train_sets_targets.pt")
torch.save((testing_sets, testing_targets), f"{data.PATH}/{ID}/test_sets_targets.pt")


model = eval(f"rnn.{constants.NET}")
net = model(data.N_ELECTRODES, constants.HIDDEN_SIZE, 7, dt=3 * dt)
dev_ = GPU if torch.cuda.is_available() else "cpu"
device_ = torch.device(dev_)
net.to(device_)

# Training
rnn.train_emg(
    net,
    ID,
    n_iterations=constants.N_ITERATIONS,
    BATCHS=constants.BATCH_SIZE,
    beta1=constants.BETA1,
    beta2=constants.BETA2,
    beta_FR=constants.BETA_FR,
    config="main",
    perc_reduction=constants.PERC_REDUCTION,
    try_checkpoint=True,
)

# Testing
print("\nTesting the network...")

# load the net state
loaded_checkpoint = rnn.load_checkpoint("main", ID)
train_loss = loaded_checkpoint["training_losses"][-1]
step = loaded_checkpoint["input_step"]
net.load_state_dict(loaded_checkpoint["model_state"])
net.to(torch.device(rnn.GPU if torch.cuda.is_available() else "cpu"))

pred, activity = net(
    torch.fliplr(torch.rot90(testing_sets, k=-1)).to(device_, non_blocking=True)
)

# Figures
fig, ax = plt.subplots(1, 1)
plt.plot(
    np.linspace(0, 1000, testing_targets.shape[1]),
    testing_targets.cpu()[0, :, 0],
    "x",
    label="true Hip",
    color="red",
)
plt.plot(
    np.linspace(0, 1000, testing_targets.shape[1]),
    testing_targets.cpu()[0, :, 1],
    "x",
    label="true Knee",
    color="green",
)
plt.plot(
    np.linspace(0, 1000, testing_targets.shape[1]),
    pred.detach().cpu()[:, 0, 0],
    label="pred Hip",
    color="red",
)
plt.plot(
    np.linspace(0, 1000, testing_targets.shape[1]),
    pred.detach().cpu()[:, 0, 1],
    label="pred Knee",
    color="green",
)
plt.legend()
plt.savefig("kinkin.png")

fig, ax = plt.subplots(1, 1)
for i in range(6):
    plt.plot(
        np.linspace(0, 1000, activity.shape[0]),
        activity[:, 0, i].detach().cpu() + 0.2 * i,
        linewidth=1,
    )
plt.savefig("acti.png")

print(f"\n\nTrain loss: {train_loss:0.4f} ")  # | Test loss: {test_loss:0.4f}\n\n")
