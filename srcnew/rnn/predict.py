import torch 
import torch.nn as nn
from typing import Tuple


def predict(net: nn.Module, inputs:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
    """
    predict network output

    :param net: artificial net 
    :param inputs: net input tensor of size(seq_length, data_size, input_size)
    :return: predictions tensor
    """  
    inputs = torch.fliplr(inputs.rot90(k=-1))
    with torch.no_grad():
        pred, activity = net(inputs)
    return pred, activity
