from typing import List, Union

import torch
import _pickle as cPickle

import rnn


def load_data(set_type, ID):
    print(f'{rnn.PATH}/{ID}/{set_type}_sets_targets.pt')
    return torch.load(f'{rnn.PATH}/{ID}/{set_type}_sets_targets.pt')


def load_checkpoint(config: Union[str,int], ID):
    loaded_checkpoint = torch.load(f'{rnn.PATH}/{ID}/checkpoint/checkpoint_{config}.tar',map_location=torch.device(rnn.GPU if torch.cuda.is_available() else "cpu"))
    return loaded_checkpoint


def dump_checkpoint(checkpoint: dict, config: Union[str, int], ID)->None:
    torch.save(checkpoint, f'{rnn.PATH}/{ID}/checkpoint/checkpoint_{config}.tar')


def dump_loss(comparison_losses: List[float], loss_type: str, training_config: Union[str, int], ID)->None:
    cPickle.dump(comparison_losses,open(f"{rnn.PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{(' ' + str(training_config)) if training_config!='main' else ''}/{loss_type}ing_loss.pkl", "wb" ))
