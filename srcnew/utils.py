import os

import numpy as np 
import pandas as pd 

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt 
import matplotlib.axes
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


from typing import List, Tuple
from data import PATH, N_ELECTRODES, MUSCLES

COLOR_MUSCLES = ['#FA525B', '#A03C60', '#452666', '#2B668B', '#1DB7B6','#5BD68C','#FA525B', '#A03C60', '#452666', '#2B668B', '#1DB7B6','#5BD68C']

PROP_ELEC = 0.175
SIDE_OFFSET_ELEC = 0.0825
ELECTRODE_POSITIONS_ELEC = { 
    # horizontal x vertical
    0 : (0.5,1*PROP_ELEC),  # elect 1
    1 : (0.5,0),  # elect 2 (0,-1)
    2 : (0.437,4*PROP_ELEC+SIDE_OFFSET_ELEC),  # elect 3 #0.437
    3 : (0.3045,4*PROP_ELEC+SIDE_OFFSET_ELEC),  # elect 4 #0.3045
    4 : (0.3045,3*PROP_ELEC+SIDE_OFFSET_ELEC),  # elect 5
    5 : (0.3045,2*PROP_ELEC+SIDE_OFFSET_ELEC),  # elect 6
    6 : (0.3045,1*PROP_ELEC+SIDE_OFFSET_ELEC),  # elect 7
    7 : (0.3045,0*PROP_ELEC+SIDE_OFFSET_ELEC),  # elect 8
    8 : (1-0.437,4*PROP_ELEC+SIDE_OFFSET_ELEC),  # elect 9
    9 : (1-0.3045,4*PROP_ELEC+SIDE_OFFSET_ELEC),  # elect 10
    10 : (1-0.3045,3*PROP_ELEC+SIDE_OFFSET_ELEC),  # elect 11
    11 : (1-0.3045,2*PROP_ELEC+SIDE_OFFSET_ELEC),  # elect 12
    12 : (1-0.3045,1*PROP_ELEC+SIDE_OFFSET_ELEC),  # elect 13
    13 : (1-0.3045,0*PROP_ELEC+SIDE_OFFSET_ELEC),  # elect 14
    14 : (0.5,3*PROP_ELEC),  # elect 15
    15 : (0.5,2*PROP_ELEC),  # elect 16
    16 : (np.nan,np.nan)}


PROP_ROOT = 0.13
SIDE_OFFSET_ROOT = 0.07
ELECTRODE_POSITIONS_ROOT = { 
        # horizontal x vertical
        0 : (0.5,1*PROP_ROOT),  # elect 1
        1 : (0.5,0),  # elect 2 (0,-1)
        2 : (0.48,4*PROP_ROOT+SIDE_OFFSET_ROOT),  # elect 3 #0.437
        3 : (0.38,4*PROP_ROOT+SIDE_OFFSET_ROOT),  # elect 4 #0.3045
        4 : (0.38,3*PROP_ROOT+SIDE_OFFSET_ROOT),  # elect 5
        5 : (0.38,2*PROP_ROOT+SIDE_OFFSET_ROOT),  # elect 6
        6 : (0.38,1*PROP_ROOT+SIDE_OFFSET_ROOT),  # elect 7
        7 : (0.38,0*PROP_ROOT+SIDE_OFFSET_ROOT),  # elect 8
        8 : (1-0.437,4*PROP_ROOT+SIDE_OFFSET_ROOT),  # elect 9
        9 : (1-0.34,4*PROP_ROOT+SIDE_OFFSET_ROOT),  # elect 10
        10 : (1-0.34,3*PROP_ROOT+SIDE_OFFSET_ROOT),  # elect 11
        11 : (1-0.35,2*PROP_ROOT+SIDE_OFFSET_ROOT),  # elect 12
        12 : (1-0.37,1*PROP_ROOT+SIDE_OFFSET_ROOT),  # elect 13
        13 : (1-0.37,0*PROP_ROOT+SIDE_OFFSET_ROOT),  # elect 14
        14 : (0.5,3*PROP_ROOT),  # elect 15
        15 : (0.5,2*PROP_ROOT),  # elect 16
        16 : (np.nan,np.nan)}


def get_configs(input_features: pd.DataFrame)->Tuple[np.ndarray, pd.DataFrame]:
    _, ind_configs, reverse_configs = np.unique(input_features.drop(['Amplitude', 'Delay','Cathodes','Anodes'], axis=1), axis=0, return_index=True, return_inverse=True)

    # create configuration name list
    stim_names = {'name': [], 'sub_names':[], 'cathodes': [], 'anodes': []}
    for i_config, config in enumerate(ind_configs):
        #name = f"cath{cathode}_an{'_'.join(map(str,anodes))}_amp{input_features['Amplitude'].iloc[i_config]}_freq{input_features['Frequency'].iloc[i_config]}"
        name = f"cath{'_'.join(map(str,input_features['Cathodes'].iloc[config]))}_an{'_'.join(map(str,input_features['Anodes'].iloc[config]))}_freq{input_features['Frequency'].iloc[config]}"
        stim_names['name'].append(name)
        stim_names['cathodes'].append(input_features['Cathodes'].iloc[config])
        stim_names['anodes'].append(input_features['Anodes'].iloc[config])
        sub_input_features = input_features.iloc[np.asarray(reverse_configs==i_config),:]
        sub_names = []
        for ind in range(len(sub_input_features)):
            sub_names.append(name + f"_amp{sub_input_features['Amplitude'].iloc[ind]}_pulses{input_features['Pulses'].iloc[config]}_delay{sub_input_features['Delay'].iloc[ind]}")
        stim_names['sub_names'].append(sub_names)
    stim_names = pd.DataFrame(stim_names)

    return reverse_configs, stim_names


def make_directories(id: str, input_features: pd.DataFrame, per_config_training: bool)->Tuple[int, np.ndarray, pd.DataFrame]:
    """
    make directories needed at PATH location to store all images

    :param id: ID of the test
    :param input_features: data frame of stimulation features
    :param per_config_training: boolean saying if training is carried per configuration or all at the same time
    """ 

    #nb_configs=len(input_features)
    reverse_configs, stim_names = get_configs(input_features)

    # create directories if they do not exist 
    if not os.path.isdir(f'{PATH}/{id}'):
        os.makedirs(f'{PATH}/{id}/checkpoint')
        os.makedirs(f'{PATH}/{id}/Main-training/Heatmap')
        os.makedirs(f'{PATH}/{id}/Main-training/Alpha_distribution')
        os.makedirs(f'{PATH}/{id}/Main-training/RNN_eigenvalues_distribution')
        os.makedirs(f'{PATH}/{id}/Main-training/Results/Train/Heatmap')
        os.makedirs(f'{PATH}/{id}/Main-training/Results/Train/Prediction')
        os.makedirs(f'{PATH}/{id}/Main-training/Results/Test/Prediction')
        os.makedirs(f'{PATH}/{id}/Main-training/Results/Train/PCA')
        os.makedirs(f'{PATH}/{id}/Main-training/Results/Train/E-I_activity_of_hidden_neurons')
        os.makedirs(f'{PATH}/{id}/Main-training/Results/Train/Alpha_dependant_activity_of_hidden_neurons')

        for name in stim_names['name']:
            os.makedirs(f'{PATH}/{id}/Main-training/Prediction/{name}')
            os.makedirs(f'{PATH}/{id}/Main-training/PCA/{name}')
            os.makedirs(f'{PATH}/{id}/Main-training/E-I_activity_of_hidden_neurons/{name}')
            os.makedirs(f'{PATH}/{id}/Main-training/Alpha_dependant_activity_of_hidden_neurons/{name}')
            if per_config_training:
                os.makedirs(f'{PATH}/{id}/Sub-training {name}/Prediction')
                os.makedirs(f'{PATH}/{id}/Sub-training {name}/PCA')
                os.makedirs(f'{PATH}/{id}/Sub-training {name}/Heatmap')
                os.makedirs(f'{PATH}/{id}/Sub-training {name}/E-I_activity_of_hidden_neurons')
                os.makedirs(f'{PATH}/{id}/Sub-training {name}/Alpha_dependant_activity_of_hidden_neurons')
                os.makedirs(f'{PATH}/{id}/Sub-training {name}/Alpha_distribution')
                os.makedirs(f'{PATH}/{id}/Sub-training {name}/RNN_eigenvalues_distribution')
                os.makedirs(f'{PATH}/{id}/Sub-training {name}/Results/Train/Prediction')
                os.makedirs(f'{PATH}/{id}/Sub-training {name}/Results/Train/Heatmap')
                os.makedirs(f'{PATH}/{id}/Sub-training {name}/Results/Train/PCA')
                os.makedirs(f'{PATH}/{id}/Sub-training {name}/Results/Train/E-I_activity_of_hidden_neurons/')
                os.makedirs(f'{PATH}/{id}/Sub-training {name}/Results/Train/Alpha_dependant_activity_of_hidden_neurons/')
                
    return reverse_configs, stim_names


def plot_electrode_activation(ax: matplotlib.axes.Axes, cathodes: int, anodes: List[int]):
    """
    include electrode activation image in ax 

    :param ax: the axes to modify 
    :param cathodes: activated cathodes
    :param anodes: activated anodes 
    """ 
        
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/electrode.png')), 'rb') as electrode_file:
        electrode_im = plt.imread(electrode_file)
    height, width = electrode_im.shape[0], electrode_im.shape[1]
    # to adapt depedending on the image
    x_offset = 4
    y_offset = 90 #165

    x_anodes, y_anodes = [], []
    for anode in anodes:
        x_anodes.append(ELECTRODE_POSITIONS_ELEC[anode][0]*width+x_offset)
        y_anodes.append(ELECTRODE_POSITIONS_ELEC[anode][1]*height+y_offset)

    x_cathodes, y_cathodes = [], []
    for cathode in cathodes:
        x_cathodes.append(ELECTRODE_POSITIONS_ELEC[cathode][0]*width+x_offset)
        y_cathodes.append(ELECTRODE_POSITIONS_ELEC[cathode][1]*height+y_offset)
 
    x_anodes, y_anodes = np.atleast_1d(x_anodes, y_anodes)
    x_cathodes, y_cathodes = np.atleast_1d(x_cathodes, y_cathodes)

    image = plt.imread(os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/anode.png')))
    # OffsetBox
    image_box = OffsetImage(image, zoom=0.15) #0.15
    for x0, y0 in zip(x_anodes, y_anodes):
        ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
        ax.add_artist(ab)

    image = plt.imread(os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/cathode.png')))
    # OffsetBox
    image_box = OffsetImage(image, zoom=0.15) #0.15
    for x0, y0 in zip(x_cathodes, y_cathodes):
        ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
        ax.add_artist(ab)

    ax.imshow(electrode_im)
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_root_activation(ax: matplotlib.axes.Axes, cathodes: int, anodes: List[int]):
    """
    include root activation image in ax 

    :param ax: the axes to modify 
    :param cathode: activated cathode
    :param anodes: activated anodes 
    """ 
    
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/roots_MR012.png')), 'rb') as electrode_file:
        electrode_im = plt.imread(electrode_file)
    height, width = electrode_im.shape[0], electrode_im.shape[1]
    # to adapt depedending on the image
    x_offset = -3
    y_offset = 210 #165

    x_anodes, y_anodes = [], []
    for anode in anodes:
        x_anodes.append(ELECTRODE_POSITIONS_ROOT[anode][0]*width+x_offset)
        y_anodes.append(ELECTRODE_POSITIONS_ROOT[anode][1]*height+y_offset)


    x_cathodes, y_cathodes = [], []
    for cathode in cathodes:
        x_cathodes.append(ELECTRODE_POSITIONS_ROOT[cathode][0]*width+x_offset)
        y_cathodes.append(ELECTRODE_POSITIONS_ROOT[cathode][1]*height+y_offset)

    x_anodes, y_anodes = np.atleast_1d(x_anodes, y_anodes)
    x_cathodes, y_cathodes = np.atleast_1d(x_cathodes, y_cathodes)
    print(x_cathodes, y_cathodes)
   
    image = plt.imread(os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/anode.png')))    
    # OffsetBox
    image_box = OffsetImage(image, zoom=0.08)
    for x0, y0 in zip(x_anodes, y_anodes):
        ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
        ax.add_artist(ab)   

    image = plt.imread(os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/cathode.png')))
    # OffsetBox
    image_box = OffsetImage(image, zoom=0.15)
    for x0, y0 in zip(x_cathodes, y_cathodes):
        ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
        ax.add_artist(ab)

    ax.imshow(electrode_im)
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_root_neurons(ax: matplotlib.axes.Axes, positions_list, target_list):
    """
    include root activation image in ax 

    :param ax: the axes to modify 
    :param cathode: activated cathode
    :param anodes: activated anodes 
    """ 
    
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/electrode.png')), 'rb') as electrode_file:
        electrode_im = plt.imread(electrode_file)
    height, width = electrode_im.shape[0], electrode_im.shape[1]
    # to adapt depedending on the image
    # x_offset = -3
    # y_offset = 210 #165

    x_offset = 4
    y_offset = 90 #165

    positions_list[0,:] = positions_list[0,:]*width + x_offset 
    positions_list[1,:] = positions_list[1,:]*height + y_offset 
  
    x_positions, y_positions = np.atleast_1d(positions_list[0,:] , positions_list[1,:])
    
    #print(x_positions)
    #print(y_positions)
   
    color_list = []
    for i in range(len(target_list)):
        color_list.append(COLOR_MUSCLES[target_list[i]])

    ax.scatter(x_positions, y_positions, c=color_list, marker='H', s=10)
    
    handles = []
    for i in range(len(COLOR_MUSCLES)//2):
        handles.append(mlines.Line2D([], [], color=COLOR_MUSCLES[i], marker='H', markersize=5, label=MUSCLES[i][1:], ls=''))
   
    ax.legend(handles = handles, frameon=False, edgecolor=None, loc='upper right', bbox_to_anchor=(0, 1), fontsize='xx-large')
    ax.imshow(electrode_im)
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return handles