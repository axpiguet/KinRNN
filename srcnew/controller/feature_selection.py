import numpy as np
from data import MUSCLES


def get_muscle_response(emg_array, reducted_axis):
    MaxAbs = np.nanmax(np.abs(emg_array), axis=reducted_axis)   
    return MaxAbs 


def get_SI(emg_array, muscle: str, reducted_axis=0):
    muscle_ind = MUSCLES.index(muscle)
    MaxAbs = get_muscle_response(emg_array, reducted_axis)
    SI_muscle = MaxAbs[:,muscle_ind]/(1+np.sum(MaxAbs, axis=1, where=np.asarray(muscle!=np.array(MUSCLES))))
    return SI_muscle





