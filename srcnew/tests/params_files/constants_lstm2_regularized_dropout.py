# RNN
DATA_FEATURES = {'Frequency': [20, 40, 60, 80], 
                'Pulses': [1],
                'Amplitude': [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5],
                'Session' : ["'mai2021'"]
                }
LR = 0.0001
N_ITERATIONS = 4000
BATCH_SIZE = 32
HIDDEN_SIZE = 500
BETA1 = 100*10**(-6)
BETA2 = 100*10**(-4) 
BETA_FR = None
PERC_REDUCTION=0.4
DATA_AUGMENTATION = True
SYMMETRY = True
PRE_NORMALIZATION = 'mean'
PER_CONFIG = False
NET= 'LSTM2'