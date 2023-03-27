from typing import List, Tuple
from scipy.signal import butter, sosfilt, sosfiltfilt, resample, hilbert
import numpy as np
import pandas as pd
from data import MUSCLE_OFFSETS, N_ELECTRODES, MUSCLES, FS
from numpy import nan
from scipy import signal
from scipy.signal import find_peaks

def load(conditions: dict, data: pd.DataFrame, muscles: List[str], fs:float = 5000, fs_vicon:float = 2000)->Tuple[np.ndarray, pd.DataFrame]:
    '''
    Define an EMG table which columns are muscles for nb_time time steps raws experiments corresponding to conditions
    Reduced with the max response when many raws corresponds to the same condition
    :param n:   conditions : dictionnary wich keys are columns name in subjectID_Frequency_Infos.xlsx
                and values a list of wanted raw values (String or integer)
                data: dataframe with emgs and stim features on which to apply conditions
                muscles : list of abbreviated muscle names (String)
                fs: frequency at which the signal will be resampled
                fs_vicon: actual sampling frequency of the signal
    :return:    loaded emg array
                associated conditions on the array
    '''

    # extract sub-dataframe based on conditions
    conditions_intersect ='(' + ') and ('.join([' or '.join([f"{key} == {value}" for value in values]) for key, values in conditions.items()]) + ')'
    data = data.infer_objects()
    data.query(conditions_intersect, inplace=True)
    data.sort_values(by=['Frequency', 'Amplitude', 'PulseWidth', 'Pulses'], inplace=True)
    #test

    #
    if data.empty:
        print('zuuuuut')
        return None, None
    #test
    #for muscle in muscles:
          #for ind in data.index :
             #print(eval(data[muscle].loc[ind]),muscle, ind)
    #print(float(eval(data['RSol'].loc[13].replace("[", "(").replace("]", ")"))))
    #print(data['RSol'].loc[13])

    if 'LAdd' in muscles :
        emgs = np.expand_dims(np.array(list(data[muscles].iloc[0].values)).T[0:1134,:], axis=0)
        for ind in range (1, data.shape[0]):
            emgs = np.concatenate((emgs,np.expand_dims( np.array(list(data[muscles].iloc[ind].values)).T[0:1134,:], axis=0)), axis=0)
    else :
        emgs = np.array([np.array([eval(data[muscle].loc[ind]) for ind in data.index]).T for muscle in muscles], dtype=np.float32)
        emgs = emgs.T

    # emgs are of size (n_experiments, time, n_muscles)

    stim_features = data.loc[:,:f'ElectrodeConf_{N_ELECTRODES}'] # la dataframe est déjà un dictionnaire dans le bon ordre des valeurs
    if 'PhysioScore' in stim_features.columns :
        stim_features = stim_features.drop(['PhysioScore'], axis=1)
    if 'ElectrodeConftxt' in stim_features.columns :
        stim_features = stim_features.drop(['ElectrodeConftxt'], axis=1)
    stim_features = stim_features.infer_objects()
    uniques, ind_uniques = np.unique(stim_features.drop(['Cathodes', 'Anodes'], axis=1), axis=0, return_inverse=True)

    # averge over trial if there is
    reduced_emgs = []
    reduced_stims = []
    for unique in range(len(uniques)):
        ind_scores = np.where(ind_uniques == unique)[0]
        ind_max_score = np.argmax(np.nanmax(emgs[ind_uniques==unique], axis=(1,2)))
        ind_max_score = ind_scores[ind_max_score]

        reduced_emgs.append(emgs[ind_max_score,:,:])
        reduced_stims.append(ind_max_score)

    reduced_emgs = np.array(reduced_emgs)
    stim_features = stim_features.iloc[reduced_stims]

    # resample to fs
    reduced_emgs = resample(reduced_emgs, int(fs*emgs.shape[1]/fs_vicon), axis=1)
    return reduced_emgs, stim_features



def filter(emg_array: np.ndarray, fs: float=5000, lowcut: float=10, highcut: float=450, order: int=4)->np.ndarray:
    """
    apply band-pass butterworth filter on emg_array

    :param emg_array: array to filter
    :param fs: sampling frequency in Hz
    :param lowcut: lowpass frequency in Hz
    :param highcut: highcut frequency in Hz
    :param order: order of butterworth filter
    :return: filtered emg array
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    data_filtered = sosfilt(sos, emg_array, axis=1)
    return data_filtered


def get_envelope(femg_array: np.ndarray, fs: float=5000, lowcuts: List[float]=[10,450], order: int=4)->np.ndarray:
    """
    apply low-pass butterworth filter on filtered emg array to get its envelope

    :param femg_array: array to filter
    :param fs: sampling frequency in Hz
    :param lowcut: lowpass frequency in Hz
    :param order: order of butterworth filter
    :return: envelope of rectified emg array
    """
    rectified_emg = abs(femg_array)
    envelope = np.zeros_like(femg_array)
    for config, lowcut in enumerate(lowcuts):
        lowcut = lowcut/(fs/2)
        sos = butter(order, lowcut, btype='lowpass', output='sos')
        envelope[:,config,:] = sosfiltfilt(sos, rectified_emg[:,config,:], axis=1)
    return envelope


def hilbert_envelope(femg_array: np.ndarray)->np.ndarray:
    """
    performs hilbert transformation of the data to get an envelope

    :param femg_array: array to filter
    :return: envelope of rectified emg array
    """
    analytic_signal = hilbert(femg_array, axis=1)
    envelope = np.abs(analytic_signal)
    return envelope

def realign(femg_array: np.ndarray, stim_arrays: np.ndarray)->np.ndarray:
    n_samples = femg_array.shape[1]
    femg_array_realigned = np.zeros_like(femg_array)
    for experiment in range(len(femg_array)):
     
        cathode_index = np.argmax(np.max(stim_arrays[experiment,-1,:,:], axis=0)) # find the cathode (or max of cathodes)
        factor_stim = np.correlate(stim_arrays[experiment,-1,:,cathode_index], stim_arrays[experiment,-1,:,cathode_index], mode='same')[int(n_samples/2)]
        if factor_stim != 0:
            for muscle in range(len(MUSCLES)):
                factor_emg = np.correlate(femg_array[experiment,:,muscle], femg_array[experiment,:,muscle], mode='same')[int(n_samples/2)]
                
                cross_cor = np.correlate(stim_arrays[experiment,-1,:,cathode_index], femg_array[experiment,:,muscle], mode='same') / np.sqrt(factor_stim * factor_emg)
                delay_arr = np.linspace(-0.5*n_samples, 0.5*n_samples, n_samples)
                delay = delay_arr[np.argmax(cross_cor)] + MUSCLE_OFFSETS[muscle]
                femg_array_realigned[experiment,:, muscle] = np.roll(femg_array[experiment, :, muscle], int(delay))
    return femg_array_realigned

def realign_stim(femg_array: np.ndarray, stim_arrays: np.ndarray)->np.ndarray:
    femg_array_ = signal.resample(femg_array, num=stim_arrays.shape[2], axis = 2)#(14, length)
    n_samples = femg_array_.shape[2]
   
    stim_realigned = np.zeros_like(stim_arrays)
    for experiment in range(len(femg_array_)):
        cathode_index = np.argmax(np.max(stim_arrays[experiment,0,:,:], axis=0)) # find the cathode (or max of cathodes)
        factor_stim = np.correlate(stim_arrays[experiment,0,:,cathode_index], stim_arrays[experiment,0,:,cathode_index], mode='same')[int(n_samples/2)]
        if factor_stim != 0:
            for muscle in range(len(MUSCLES)):
                factor_emg = np.correlate(femg_array_[experiment,muscle, :], femg_array_[experiment,muscle, : ], mode='same')[int(n_samples/2)]
               
                cross_cor = np.correlate(stim_arrays[experiment,0,:,cathode_index], femg_array_[experiment,muscle,:], mode='same') / np.sqrt(factor_stim * factor_emg)
                delay_arr = np.linspace(-0.5*n_samples, 0.5*n_samples, n_samples)
                delay = delay_arr[np.argmax(cross_cor)] + MUSCLE_OFFSETS[muscle]
               
                stim_realigned[experiment,:,:,:] = np.roll(stim_arrays[experiment,:,:,:], -40*int(delay), axis =2)
    return stim_realigned

def realign_stimUnit(femg_array: np.ndarray, stim_arrays: np.ndarray)->np.ndarray:      # emg = length x14     stim = length x 17
    femg_array_ = signal.resample(femg_array, num=stim_arrays.shape[0], axis = 0)
    n_samples = femg_array_.shape[0]
    print('this should be big ' , n_samples)
    stim_realigned = np.zeros_like(stim_arrays)
    for experiment in range(1):
        cathode_index = np.argmax(np.max(stim_arrays, axis=0)) # find the cathode (or max of cathodes)
        print('cathode ',cathode_index)
        factor_stim = np.correlate(stim_arrays[:,cathode_index], stim_arrays[:,cathode_index], mode='same')[int(n_samples/2)]
        if factor_stim != 0:
            max = 0
            chosen = None
            delay = 0
            for muscle in range(len(MUSCLES)):
                factor_emg = np.correlate(femg_array_[:,muscle], femg_array_[:,muscle], mode='same')[int(n_samples/2)]
                if np.max(femg_array_[:,muscle]) > max:
                    max = np.max(femg_array_[:,muscle])
                    chosen = muscle
                if muscle == chosen :
                    print(chosen)
                    cross_cor = np.correlate(stim_arrays[:,cathode_index], femg_array_[:,muscle], mode='same') / np.sqrt(factor_stim * factor_emg)
                    delay_arr = np.linspace(-0.5*n_samples, 0.5*n_samples, n_samples)
                    delay = delay_arr[np.argmax(cross_cor)] + MUSCLE_OFFSETS[muscle]
                    
            stim_realigned = np.roll(stim_arrays, -2*int(delay), axis =0)
    return stim_realigned, -2*int(delay)


def rolling(stim_arrays: np.ndarray,  triggerEMG :float)->np.ndarray: # stim = length x 17
    cathode_index = np.argmax(np.max(stim_arrays, axis=0))
    thresh = np.max(stim_arrays[:,cathode_index])/2
    peaks = find_peaks(stim_arrays[:,cathode_index], height = thresh)
    
    begin = peaks[0][0]
    return np.roll(stim_arrays, -int(begin-triggerEMG+ 2.5*np.mean(MUSCLE_OFFSETS)), axis =0), -int(begin-triggerEMG+ 2.5*np.mean(MUSCLE_OFFSETS))
