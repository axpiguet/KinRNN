# RNN
DATA_FEATURES = {'Frequency': [20, 40, 80],
                'Pulses': [1],
                'Amplitude': [2.5,3,3.5,4,4.5],
                'Session' : ["'mai2021'"]
                }
LR = 1#0.0001
N_ITERATIONS = 200
BATCH_SIZE = 1
HIDDEN_SIZE = 500
BETA1 = 0.1*10**(-6)
BETA2 = 0.1*10**(-4)
BETA_FR = None
PERC_REDUCTION=0.7
DATA_AUGMENTATION = True
SYMMETRY = True
PRE_NORMALIZATION = 'mean'
PER_CONFIG = False
NET= 'LSTM2NoBiomech'
