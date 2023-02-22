# RNN
DATA_FEATURES = {'Frequency': [20], 
                'Pulses': [4],
                'Amplitude': [3],
                'Session' : ["'mai2021'"]
                }
LR = 0.0001
N_ITERATIONS = 1200
BATCH_SIZE = 32
HIDDEN_SIZE = 500
BETA1 = 0.1*10**(-6)
BETA2 = 0.1*10**(-4)
BETA_FR = None
PERC_REDUCTION=0.7
DATA_AUGMENTATION = True
SYMMETRY = True
PRE_NORMALIZATION = 'mean'
PER_CONFIG = False
NET= 'RNN'