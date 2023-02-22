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




alphas = torch.load( f'{data.PATH}/{ID}/alphas.pt').cpu().detach().numpy()

from scipy import stats
x = np.linspace(np.min(alphas),np.max(alphas),300)

density = stats.gaussian_kde(alphas[:,1])
y = density(x)

fig, axs = plt.subplots(2, int(np.shape(alphas)[1]/2),figsize=(13,5))

for i in range(int(np.shape(alphas)[1]/2)):
    density = stats.gaussian_kde(alphas[:,i])
    yL = density(x)
    density = stats.gaussian_kde(alphas[:,i+int(np.shape(alphas)[1]/2)])
    yR = density(x)
    axs[0, i].plot(x, yL)
    axs[1, i].plot(x, yR)
plt.title("Density Plot of the data")
plt.savefig("alphas.png")
