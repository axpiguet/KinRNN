import numpy as np
import pandas as pd
import scipy as sp

N_ELECTRODES = 17
TAU = 300 # [µs]
PULSE_WIDTH = TAU # [µs]
WIDTH_VARIANCES_FACTORS = np.array([3/25, 2/25, 1/25, 1/50])
FS = 1000 # [Hz]


def build_pulse(middle: int, variance: float, time_array: np.ndarray)->np.ndarray:
    '''
    Build a positive square-shaped curve
    :param n:   middle : middle time of the density function
                variance : width of the density function
                time_array : stimulation time points
    :return:    stim signal with same size as time_array
    '''
    # if to high or negative, goes at the beginning or the end respectively
    if middle>=np.max(time_array):
        middle -= np.max(time_array)
    elif middle < 0:
        middle += np.max(time_array)

    # draw a gaussian and normalize it
    stim = sp.stats.norm.pdf(time_array, loc=middle, scale=0.05*variance)
    stim/=np.max(stim)
    return stim


def create(stim_features: pd.DataFrame, duration: int, fs :float=5000)->np.ndarray:
    '''
    Create the stimulation time serie for all electrodes
    :param n:   stim_features: data frame of stimulation features
                duration : time length of the stimulation [ms]
                fs: sampling frequency [Hz]
    :return:    array of (nb_variances, time length, n_experiments, n_electrodes) containing stimulation time series
    '''
    pulse_freq = 500 # [Hz]
    time_stim = np.linspace(0, duration-1, num=int(0.001*duration*fs)) # in ms
    n_experiments = len(stim_features)

    # create variances array
    variances = np.array([FS/freq for freq in stim_features.loc[:,"Frequency"]]).T
    variances = []
    for freq in stim_features.loc[:,"Frequency"]:
        variances.append(np.append(1000/freq*WIDTH_VARIANCES_FACTORS,0.5*PULSE_WIDTH*10**(-3)))
    variances = np.array(variances).T

    stim_series = np.zeros((n_experiments, len(variances),int(0.001*duration*fs), N_ELECTRODES))

    for i_variance, variance in enumerate(variances):
        for i_experiment, experiment in enumerate(stim_features.index):
            nb_burst = int(stim_features.loc[experiment,'Frequency']*0.001*duration)

            # create a range of bursts
            stim_step = np.zeros_like(time_stim)
            for burst in range(nb_burst):
                for pulse in range(stim_features.loc[experiment,'Pulses']):
                    start_ =  int(burst*fs/stim_features.loc[experiment,'Frequency'] + pulse*fs/pulse_freq)
                    end_ = start_ + int(fs*(stim_features.loc[experiment,'PulseWidth']*10**(-6))) +1
                    middle = (start_ + end_)//(2*fs*10**(-3)) + 20 # convertion in ms and add little something to not cut the stim at beginning for large variances

                    stim_step += build_pulse(middle,variance[i_experiment], time_stim)
            stim_step *= stim_features.loc[experiment,'Amplitude']

            # change the sign and magnitude according to anodes or cathodes
            for electrode in range(0,N_ELECTRODES) :
                if (electrode in stim_features.loc[experiment,'Anodes']) : # first up then down
                        totalwaveform = stim_step/len(stim_features.loc[experiment,'Anodes'])
                        stim_series[i_experiment, i_variance, :, electrode] = totalwaveform

                elif (electrode in stim_features.loc[experiment,'Cathodes']) :  # first down then up
                        totalwaveform = -stim_step/len(stim_features.loc[experiment,'Cathodes'])
                        stim_series[i_experiment, i_variance, :, electrode]= totalwaveform
                else:
                    stim_series[i_experiment, i_variance, :, electrode] = np.zeros_like(time_stim)

    stim_series = stim_series.astype('float64')
    return stim_series
