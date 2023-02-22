from os.path import dirname, abspath, join
import numpy as np

ROOT_DIR = dirname(__file__)
PATH = abspath(join(ROOT_DIR, '../../tests_results/'))

SUBJECT_ID = 'MR012'
FS = 1000 # [Hz]
#MUSCLES = ['LAdd', 'LRF', 'LVLat', 'LST', 'LMG','LTA', 'LSol', 'RAdd', 'RRF', 'RVLat', 'RST', 'RMG', 'RSol']
MUSCLES = ['LIl', 'LRF', 'LVLat', 'LST','LTA', 'LMG', 'LSol', 'RIl', 'RRF', 'RVLat', 'RST','RTA', 'RMG', 'RSol'] # LTA RTA

# Processing
ENVELOPE = False
LOWCUTS = [90, 30, 50, 1.0001]

# Stim
#MUSCLE_OFFSETS = np.array([5, 10, 10, 15, 17, 20, 25, 5, 10, 10, 15, 20, 25]) + 6
MUSCLE_OFFSETS = np.array([5, 10, 10, 15, 17, 20, 25, 5, 10, 10, 15, 17, 20, 25]) + 6 # [ms] 12
TAU = 300 # [µs]
PULSE_WIDTH = TAU # [µs]
N_ELECTRODES = 17
WIDTH_VARIANCES_FACTORS = np.array([3/25, 2/25, 1/25, 1/50])


from .dataset import augment_data, normalize, train_test_split
from .emg import load, filter, realign
from .stim import create
from .plot_data import plot_amp_stimvsresponse, plot_amplitude_modulation, plot_bars, plot_si_bars, plot_colormesh, plot_emgs , recrutement
