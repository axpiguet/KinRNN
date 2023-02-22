import math
import numpy as np
import torch
from torch.nn import init
 
 
def mean_initialization(parameters_dict: dict)->dict:
    """
    initialize weights for main training with the average of the params of the other training

    :param parameters_dict: dictionnary of parameters which keys are the parameters and values a list of this parameter at the end of each pre-training
    """ 
    new_params = {}
    for name, params in parameters_dict.items(): new_params[name] = np.mean(params, axis=0)
    return new_params


def max_initialization(parameters_dict: dict, take_alpha: bool=True)->dict:
    """
    initialize weights for main training with the absolute maximum of the params of the other training

    :param parameters_dict: dictionnary of parameters which keys are the parameters and values a list of this parameter at the end of each pre-training
    :param take_alpha: if True, alphas are initialized with highest h2h.weight
    """ 
    new_params = {}
    for name, params in parameters_dict.items(): 
        if name != 'alpha' and name != 'alpha_s':
            ind_max = np.argmax(np.abs(params), axis=0)
            new_params[name] = np.take_along_axis(np.array(params), np.expand_dims(ind_max, axis=0), axis=0).squeeze(axis=0)
            if ('hh' in name or 'h2h' in name) and 'weight' in name and take_alpha:
                configs = np.unique(ind_max)
                counts = np.array([np.count_nonzero(ind_max==config, axis=0) for config in configs])
                max_count = np.argmax(counts, axis=0)
                new_params['alpha'] = np.array([parameters_dict['alpha'][max_count[neuron]][:,neuron] for neuron in range(len(max_count))]).reshape((1,-1))
                if 'alpha_s' in parameters_dict.keys():
                    new_params['alpha_s'] = np.array([parameters_dict['alpha_s'][max_count[neuron]][:,neuron] for neuron in range(len(max_count))]).reshape((1,-1))
    return new_params


def rank_initialization(parameters_dict: dict, alpha: torch.Tensor, take_alpha: bool=True)->dict:
    """
    initialize weights for main training with the absolute maximum ranking of the params of the other training

    :param parameters_dict: dictionnary of parameters which keys are the parameters and values a list of this parameter at the end of each pre-training
    :param take_alpha: if True, alphas are initialized with highest h2h.weight
    """ 
    # initialize new params list with kaiming uniform 
    new_params = {}
    for name, params in parameters_dict.items():
        new_params[name] = torch.zeros(params[0].shape)
        if 'weight' in name:
            init.kaiming_uniform_(new_params[name])
            fan_in, _ = init._calculate_fan_in_and_fan_out(new_params[name])
        if 'bias' in name:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(new_params[name], -bound, bound)
        if name == 'alpha':
            new_params[name] = alpha.detach()
        new_params[name] = new_params[name].numpy()

        # find order of ranking 
        if name != 'alpha' and name != 'alpha_s':
            nb_elements = len(params[0].flatten())//len(params)
            config_ranking = np.array([np.argsort(np.abs(param), axis=None)+(i*len(params[0].flatten())) for i, param in enumerate(params)])
            np.put(config_ranking, config_ranking, np.arange(len(params[0].flatten())))
            max_ranking = np.argmax(config_ranking, axis=0)
            # define new parameter 
            new_params[name] = [params[connection].flatten()[i_connection] if config_ranking[connection][i_connection]>= nb_elements else new_params[name].flatten()[i_connection] for i_connection, connection in enumerate(max_ranking)]
            new_params[name] = np.reshape(new_params[name], params[0].shape)
            
            # if matrice is RNN define alpha 
            if ('hh' in name or 'h2h' in name) and 'weight' in name and take_alpha:
                config_ranking = np.reshape(config_ranking, (-1, params[0].shape[0], params[0].shape[1]))
                config_ranking = np.amax(config_ranking, axis=1)
                max_count = np.argmax(config_ranking, axis=0)
                new_params['alpha'] = np.array([parameters_dict["alpha"][connection][:,i_connection] if config_ranking[connection][i_connection]>= nb_elements else new_params['alpha'][i_connection] for i_connection, connection in enumerate(max_count)]).reshape((1,-1))
                if 'alpha_s' in parameters_dict.keys():
                    new_params['alpha_s'] = np.array([parameters_dict["alpha_s"][connection][:,i_connection] if config_ranking[connection][i_connection]>= nb_elements else new_params['alpha_s'][i_connection] for i_connection, connection in enumerate(max_count)]).reshape((1,-1))

    return new_params
 