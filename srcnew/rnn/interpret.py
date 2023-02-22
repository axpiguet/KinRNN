import os
from typing import Union
import time
import psutil

import _pickle as cPickle
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler

from sklearn.decomposition import PCA
import torch
import torch.nn as nn

from data import MUSCLES, FS, PATH
from .predict import predict
from utils import plot_electrode_activation


matplotlib.use('Agg')
plt.style.use('dark_background')


def display_net_state(optimizer: torch.optim, training_loss: float, epoch:int, testing_loss: float=None)->None:
    """
    display net state on terminal

    :param optimizer
    :param running_loss: actual loss
    :param epoch: actual epoch
    """
    optim_param = optimizer.param_groups[0]


    to_print = '{}   proc {}   epoch {}   train_loss {:0.2f}'.format(time.ctime(), os.getpid(), epoch, training_loss)
    if testing_loss is not None:
        to_print += '   test_loss {:0.2f}'.format(testing_loss)

    if 'exp_avg' in optimizer.state[optim_param['params'][1]].keys() and 'exp_avg' in optimizer.state[optim_param['params'][3]].keys() and  'exp_avg' in optimizer.state[optim_param['params'][5]].keys():
        # get averaged B step size
        state_B = optimizer.state[optim_param['params'][1]]
        unbiased_exp_avg = state_B['exp_avg']/(1-optim_param['betas'][0]**state_B['step'])
        unbiased_exp_avg_sq = state_B['exp_avg_sq']/(1-optim_param['betas'][1]**state_B['step'])
        lr_B = np.format_float_scientific(torch.mean(optim_param['lr'] / (1-optim_param['betas'][0]**state_B['step']) * unbiased_exp_avg / (torch.sqrt(unbiased_exp_avg_sq) + optim_param['eps'])).item(), precision=3)

        # get averaged J step size
        state_J = optimizer.state[optim_param['params'][3]]
        unbiased_exp_avg = state_J['exp_avg']/(1-optim_param['betas'][0]**state_J['step'])
        unbiased_exp_avg_sq = state_J['exp_avg_sq']/(1-optim_param['betas'][1]**state_J['step'])
        lr_J = np.format_float_scientific(torch.mean(optim_param['lr'] / (1-optim_param['betas'][0]**state_B['step']) * unbiased_exp_avg / (torch.sqrt(unbiased_exp_avg_sq) + optim_param['eps'])).item(), precision=3)

        # get averaged W step size
        state_W = optimizer.state[optim_param['params'][5]]
        unbiased_exp_avg = state_W['exp_avg']/(1-optim_param['betas'][0]**state_W['step'])
        unbiased_exp_avg_sq = state_W['exp_avg_sq']/(1-optim_param['betas'][1]**state_W['step'])
        lr_W = np.format_float_scientific(torch.mean(optim_param['lr'] / (1-optim_param['betas'][0]**state_B['step']) * unbiased_exp_avg / (torch.sqrt(unbiased_exp_avg_sq) + optim_param['eps'])).item(), precision=3)

        to_print += '   lrB {}   lrJ {}   lrW {}'.format(lr_B, lr_J, lr_W)

    to_print += '   RAM {}%'.format(psutil.virtual_memory()[2])
    print(to_print)

def plot(net: nn.Module, inputs:torch.Tensor, labels: torch.Tensor, input_names: pd.DataFrame, reverse_configs: np.ndarray, device: torch.device, ID, training_config: Union[str, int]='main', epoch=None, split=None)->None:
    """
    call all necessary plot functions

    :param net
    :param inputs: inputs of the network of size (time, batch_size, n_electrodes)
    :param labels: labels of the inputs of size (time, batch_size, n_muscles)
    :param input_names: dictionnary with key name, sub_names, cathode and anodes
    :param reverse_configs: numpy array output of numpy.unique function with return_inverse=True
    :param epoch: actual epoch
    :param device: actual device cuda or cpu
    :param training_config: sub training or main training
    """
    pred, activity = predict(net, inputs.to(device))
    print('Plotting : ', pred.shape)
    inputs, labels, pred, activity = inputs.numpy(), labels.numpy(), pred.cpu().numpy(), activity.cpu().numpy()

    # shape inputs and labels per configuration
    inputs_per_config, labels_per_config, pred_per_config, activity_per_config = [], [], [], []
    for config in range(len(input_names)):
        inputs_per_config.append(inputs[np.asarray(config==reverse_configs),:,:])
        labels_per_config.append(labels[np.asarray(config==reverse_configs),:,:])
        pred_per_config.append(pred[:,np.asarray(config==reverse_configs),:])  # pred and activity are of different shape
        activity_per_config.append(activity[:,np.asarray(config==reverse_configs),:])

    plot_pred(inputs_per_config, labels_per_config, pred_per_config, input_names, ID, training_config=training_config, epoch=epoch, split=split)

    if split != 'Test':
        weight_dict = net.get_weights()
        for (weight_name, weight) in weight_dict.items():
            plot_heatmap(weight, weight_name, ID, training_config=training_config, epoch=epoch)
        plot_EI_activity(net, activity_per_config, input_names, ID, training_config=training_config, epoch=epoch)
        #alpha_hist_n = plot_alpha_distribution(net.alpha, ID, training_config=training_config, epoch=epoch)
        #plot_alpha_activity(net, alpha_hist_n, activity_per_config, input_names, ID, training_config=training_config, epoch=epoch)
        plot_PCA(activity_per_config, input_names, ID, training_config=training_config, epoch=epoch)
        plot_eigenvalues(net, ID, training_config=training_config, epoch=epoch)


def plot_pred(inputs: np.ndarray, labels: np.ndarray, pred: np.ndarray, input_names: pd.DataFrame, ID, training_config: Union[str, int]='main', epoch=None, split='Train', other_path: str=None)->None:
    """
    apply low-pass butterworth filter on filtered emg array to get its envelope

    :param inputs: inputs of the network of size (config, time, n_sub_config, n_electrodes)
    :param labels: labels of the inputs of size (congig, time, n_sub_config, n_muscles)
    :param pred: prediction of the net when giving the inputs of size (congig, time, n_sub_config, n_muscles)
    :param input_names: dictionnary with key name, sub_names, cathode and anodes
    :param epoch: actual epoch
    :param training_config: sub training or main training
    """
    time_stim = np.linspace(0, inputs[0].shape[1]*1000//FS-1, num=inputs[0].shape[1])
    n_muscles = len(MUSCLES)//2

    for i_config in range(len(input_names)):
        for i_sub_config in range(inputs[i_config].shape[0]): # number of sub-configurations
            fig = plt.figure(figsize=(40,15)) # 30 15
            spec = gridspec.GridSpec(pred[i_config].shape[2]//2+pred[i_config].shape[2]%2+1,3, width_ratios=[0.1,0.45,0.45])
            axs=[]
            for i in range(len(MUSCLES)//2+len(MUSCLES)%2+1):
                raw = []
                for j in range(1,3):
                    raw.append(fig.add_subplot(spec[i, j]))
                axs.append(raw)
            axs=np.array(axs)

            axs[0,0].set_ylabel('stim', fontsize="40", color = 'k')
            axs[0,0].tick_params('y',labelsize="30",color = 'k')
            axs[0,1].tick_params('y', labelleft=False,color = 'k')
            axs[0,0].tick_params('x', labelbottom=False, bottom=False,color = 'k')
            axs[0,1].tick_params('x', labelbottom=False, bottom=False,color = 'k')
            #axs[0,0].ticklabel_format(axis='y', style='sci')

            for c in range(2):
                axs[0,c].plot(time_stim, inputs[i_config][i_sub_config,:,:], color='#fa525b')
                axs[0,c].set_frame_on(False)
                axs[0,c].set_title('RIGHT' if c else 'LEFT', fontsize='50',color = 'k')
                axs[0,c].set_ylim(-100,100)

                indexmusc = range(1,len(MUSCLES)//2+len(MUSCLES)%2+1)
                if c and (len(MUSCLES) == 13) and (MUSCLES[0] == 'LAdd'):
                    indexmusc = [1,2,3,4,5,7]
                #for r in range(1,len(MUSCLES)//2+len(MUSCLES)%2+1):
                for r in indexmusc:
                    if labels is not None: expected_line = axs[r,c].plot(time_stim, labels[i_config][i_sub_config,:,c*n_muscles + r-1], '-',color= 'darkgrey', linewidth=4)  #'#313335')'#959798'
                    predicted_line = axs[r,c].plot(time_stim, pred[i_config][:,i_sub_config,c*n_muscles + r-1], '--' , color='dimgrey', linewidth=4) ##53555a  #dbdbdd
                    #if labels is not None: expected_line = axs[r,c].plot(time_stim, labels[i_config][i_sub_config,:,c*n_muscles + r-1], '-',color= 'darkgrey', linewidth=4)  #'#313335')'#959798'
                    axs[r,c].set_frame_on(False)
                    #axs[r,c].set_ylim(-100,100)
                    axs[r,c].set_ylim(-50,50)
                    axs[r,c].tick_params('x', labelbottom=False, color = 'k')
                axs[6,c].set_frame_on(False)
                axs[6,c].set_ylim(-50,50)
                axs[6,c].tick_params('x', labelbottom=False, color = 'k')

                axs[-1,c].tick_params('x', labelbottom=True, labelsize="30", color = 'k')
                #axs[-1,c].ticklabel_format(axis='x', style='sci')

                axs[-1,c].set_xlabel('Time (ms)', fontsize="40", color = 'k')

            for r in range(1,len(MUSCLES)//2+len(MUSCLES)%2+1):
                axs[r,0].set_ylabel(f'{MUSCLES[r-1][1:]}', fontsize="40" , color = 'k')
                axs[r,0].tick_params('y',labelsize="30",color = 'k')
                axs[r,1].tick_params('y', labelleft=False,color = 'k')
                #axs[r,0].ticklabel_format(axis='y', style='sci')

            ax = fig.add_subplot(spec[:,0])
            plot_electrode_activation(ax, input_names['cathodes'].iloc[i_config], input_names['anodes'].iloc[i_config])
            #plt.legend(handles=[predicted_line, expected_line], labels=['Predicted', 'Expected'], frameon=False, fontsize='xx-large')
            plt.subplots_adjust(top=0.88, bottom=0.08, left=0, right=0.985, hspace=0.9, wspace=0.1)
            if split is not None:
                #plt.savefig(f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{' ' + input_names['name'].iloc[i_config] if training_config!='main' else ''}/Results/{split}/Prediction/{input_names['sub_names'].iloc[i_config][i_sub_config]}.svg", transparent=True,format='svg', dpi=1200)
                plt.savefig(f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{' ' + input_names['name'].iloc[i_config] if training_config!='main' else ''}/Results/{split}/Prediction/{input_names['sub_names'].iloc[i_config][i_sub_config]}.png", transparent=True)
            elif other_path is not None:
                #plt.savefig(f"{other_path}/test_{input_names['sub_names'].iloc[i_config][i_sub_config]}.svg", transparent=True,format='svg', dpi=1200)
                plt.savefig(f"{other_path}/test_{input_names['sub_names'].iloc[i_config][i_sub_config]}.png", transparent=True)
            else:
                #plt.savefig(f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{' ' + input_names['name'].iloc[i_config] if training_config!='main' else ''}/Prediction/{input_names['name'].iloc[i_config] + '/' if training_config=='main' else ''}{input_names['sub_names'].iloc[i_config][i_sub_config]}_{epoch}.svg", transparent=True,format='svg', dpi=1200)
                plt.savefig(f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{' ' + input_names['name'].iloc[i_config] if training_config!='main' else ''}/Prediction/{input_names['name'].iloc[i_config] + '/' if training_config=='main' else ''}{input_names['sub_names'].iloc[i_config][i_sub_config]}_{epoch}.png", transparent=True)

            #plt.close(fig)


def plot_EI_activity(net: nn.Module, activity: np.ndarray, input_names: pd.DataFrame, ID, training_config: Union[str, int]='main', epoch=None) -> None:
    """
    plot Excitatory-Inhibitory activity of the network

    :param net
    :param activity: activity of the network of size (config, time, n_sub_config, hidden_size)
    :param input_names: dictionnary with key name, sub_names, cathode and anodes
    :param epoch: actual epoch
    :param training_config: sub training or main training
    """

    rnn_weights = net.get_weights()["J"].cpu().detach().numpy()
    time_stim = np.linspace(0, activity[0].shape[0]*1000//FS-1, num=activity[0].shape[0])

    mean_activity = np.mean(rnn_weights, axis=0)
    ind_sort = np.argsort(-mean_activity)
    ordered_activity = []
    for i in range(len(activity)):
        ordered_activity.append(activity[i][:,:,ind_sort])

    wlim = np.percentile(np.abs(rnn_weights),99)
    wlim = int(wlim*100)/100
    step=25
    color_map = plt.cm.RdBu_r(np.linspace(0, 1, activity[0].shape[2]//step))

    for i_config in range(len(input_names['name'])):
        for i_sub_config in range(activity[i_config].shape[1]):
            fig = plt.figure(figsize=(20,5))
            matplotlib.rcParams['axes.prop_cycle'] = cycler(color=list(color_map))
            plt.plot(time_stim, ordered_activity[i_config][:,i_sub_config,::step])
            plt.xlabel('time (ms)')
            plt.ylabel('E-I Activity')
            plt.box(False)
            cax = fig.add_axes([0.9, 0.1, 0.02, 0.7])
            cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-wlim, vmax=wlim), cmap='RdBu_r'),cax=cax, ticks=[-wlim, 0, wlim])
            cbar.set_label("J weight", labelpad=-0.2)
            cbar.outline.set_linewidth(0)
            if epoch is None:
                plt.savefig(f"{PATH}/{ID}/Main-training/Results/Train/E-I_activity_of_hidden_neurons/{input_names['sub_names'].iloc[i_config][i_sub_config]}.png", transparent=True)
            else:
                plt.savefig(f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{' ' + input_names['name'].iloc[i_config] if training_config!='main' else ''}/E-I_activity_of_hidden_neurons/{input_names['name'].iloc[i_config] + '/' if training_config=='main' else ''}{input_names['sub_names'].iloc[i_config][i_sub_config]}_{epoch}.png", transparent=True)
            plt.close(fig)

    matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])


def plot_alpha_activity(net: nn.Module, alpha_hist_n: np.ndarray, activity: np.ndarray, input_names: pd.DataFrame, ID, training_config: Union[str, int]='main', epoch=None) -> None:
    """

    Plot alpha activity of hidden neurons

    :param net
    :param alpha_hist_n: array that counts alphas for each value
    :param activity: activity of the network of size (config, time, n_sub_config, hidden_size)
    :param input_names: dictionnary with key name, sub_names, cathode and anodes
    :param epoch: actual epoch
    :param training_config: sub training or main training
    """

    alphas = net.alpha[0,:].cpu().detach().numpy()
    time_stim = np.linspace(0, activity[0].shape[0]*1000//FS-1, num=activity[0].shape[0])

    ind_sort = np.argsort(-alphas)
    ordered_activity = []
    for i in range(len(activity)):
        ordered_activity.append(activity[i][:,:,ind_sort])

    step=25
    color_map = plt.cm.RdBu_r
    color_list=[]
    for j in range(len(alpha_hist_n)):
        for i in range(int(alpha_hist_n[j])):
            color_list.append(np.array(color_map(j/(alphas.shape[0]//10))))


    matplotlib.rcParams['axes.prop_cycle'] = cycler(color=color_list[::step])
    wmax = np.percentile(np.max(alphas),99)
    wmax = int(wmax*100)/100

    for i_config in range(len(input_names['name'])):
        for i_sub_config in range(activity[i_config].shape[1]):
            fig = plt.figure(figsize=(20,5))
            plt.plot(time_stim, ordered_activity[i_config][:,i_sub_config,::step])
            plt.xlabel('time (ms)')
            plt.ylabel('Alpha-dependant activity')
            plt.box(False)
            cax = fig.add_axes([0.9, 0.1, 0.02, 0.7])
            cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-wmax, vmax=wmax), cmap='RdBu_r'),cax=cax, ticks=[-wmax, 0, wmax])
            cbar.ax.set_yticks(cbar.ax.get_yticks(), labels=np.round(np.linspace(np.min(alphas), np.max(alphas),cbar.ax.get_yticks().shape[0]), decimals=2))
            cbar.set_label("Alpha", labelpad=-0.2)
            cbar.outline.set_linewidth(0)
            if epoch is None:
                plt.savefig(f"{PATH}/{ID}/Main-training/Results/Train/Alpha_dependant_activity_of_hidden_neurons/{input_names['sub_names'].iloc[i_config][i_sub_config]}.png", transparent=True)
            else:
                plt.savefig(f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{' ' + input_names['name'].iloc[i_config] if training_config!='main' else ''}/Alpha_dependant_activity_of_hidden_neurons/{input_names['name'].iloc[i_config] + '/' if training_config=='main' else ''}{input_names['sub_names'].iloc[i_config][i_sub_config]}_{epoch}.png", transparent=True)
            plt.close(fig)

    matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])


def plot_heatmap(weight_tensor: torch.Tensor, weight_name: str, ID, training_config: Union[str, int]='main', epoch=None)-> None:
    """
    plot heatmap of weights

    :param weight_tensor: weight matrix of any size
    :param weight_name: weight name
    :param epoch: epoch
    :param training_config: int defining configuration or String 'main'
    """
    weight_tensor = weight_tensor.cpu().detach().numpy()
    fig = plt.figure(figsize=(6,5))
    wlim = np.percentile(np.abs(weight_tensor),99)
    #wlim = 0.15
    wlim = int(wlim*100)/100
    ax=fig.add_axes([0.1,0.1,0.7,0.7])

    im = ax.imshow(weight_tensor, cmap='RdBu_r', vmin=-wlim, vmax=wlim)

    plt.xlabel('From neurons')
    plt.ylabel('To neurons')
    for loc in ['left', 'right', 'top', 'bottom']:
    # ax.spines[loc].set_color('gray')
        ax.spines[loc].set_visible(False)
    divider = make_axes_locatable(ax)
    # Create colorbar
    cax = fig.add_axes([0.82, 0.1, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cax, ticks=[-wlim, 0, wlim])
    cbar.set_label(f"{weight_name} connection", labelpad=-1)
    cbar.outline.set_linewidth(0)
    # Rotate the tick labels and set their alignment.
    #plt.setp(plt.xticks()[1], rotation=0, ha="center",rotation_mode="anchor")

    # Turn spines off and create white grid.
    if epoch is None:
        plt.savefig(f"{PATH}/{ID}/Main-training/Results/Train/Heatmap/{weight_name}.png", transparent=True)
    else:
        plt.savefig(f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{' ' + training_config if training_config!='main' else ''}/Heatmap/{weight_name}_{epoch}.png", transparent=True)
    plt.close(fig)


def plot_alpha_distribution(alpha: torch.Tensor, ID, training_config: Union[str, int]='main', epoch=None)->Figure:
    """
    plot alpha distribution in histogram

    :param alpha: alpha tensor
    :param epoch: actual epoch
    :param training_config: sub training or main training
    """
    alpha = alpha.cpu().detach().numpy()

    color_map = plt.cm.RdBu_r
    n, _, _ = plt.hist(alpha.T, alpha.shape[1]//10, density=False)

    fig = plt.figure()
    _, bins, patches = plt.hist(alpha.T, alpha.shape[1]//10, density=True)
    for i, p in enumerate(patches):
        plt.setp(p, 'facecolor', color_map(i/(alpha.shape[1]//10))) # notice the i/25
    plt.xlabel("Alpha")
    plt.ylabel("Distribution in neurons")
    plt.box(False)
    if epoch is None:
        plt.savefig(f"{PATH}/{ID}/Main-training/Results/Train/alpha_distribution.png", transparent=True)
    else:
        plt.savefig(f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{' ' + training_config if training_config!='main' else ''}/Alpha_distribution/{epoch}.png", transparent=True)
    plt.close(fig)

    return n


def plot_PCA(activity: np.ndarray, input_names: pd.DataFrame, ID, training_config: Union[str, int]='main', epoch=None)->None:
    """
    plot PCA two first components of net activity

    :param activity: activity of the network of size (config, time, n_sub_config, hidden_size)
    :param input_names: dictionnary with key name, sub_names, cathode and anodes
    :param epoch: actual epoch
    :param training_config: sub training or main training
    """

    time_stim = np.linspace(0, activity[0].shape[0]*1000//FS-1, num=activity[0].shape[0])

    for i_config in range(len(input_names['name'])):
        for i_sub_config in range(activity[i_config].shape[1]):
            pca = PCA(n_components=2)
            pca.fit(activity[i_config][:,i_sub_config,:])
            activity_pc = pca.transform(activity[i_config][:,i_sub_config,:])

            fig = plt.figure()
            plt.scatter(activity_pc[:, 0], activity_pc[:, 1], s=5, c=time_stim, marker='o', cmap='twilight', alpha=1)
            plt.xlabel('PC activity 1')
            plt.ylabel('PC activity 2')
            plt.box(False)
            cax = fig.add_axes([0.9, 0.1, 0.02, 0.7])
            cbar = plt.colorbar(cax=cax, ticks=[0, activity[i_config].shape[0]])
            cbar.outline.set_linewidth(0)
            cbar.set_label("time (ms)", labelpad=-1)
            if epoch is None:
                plt.savefig(f"{PATH}/{ID}/Main-training/Results/Train/PCA/{input_names['sub_names'].iloc[i_config][i_sub_config]}.png", transparent=True)
            else:
                plt.savefig(f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{' ' + input_names['name'].iloc[i_config] if training_config!='main' else ''}/PCA/{input_names['name'].iloc[i_config] + '/' if training_config=='main' else ''}{input_names['sub_names'].iloc[i_config][i_sub_config]}_{epoch}.png", transparent=True)
            plt.close(fig)


def plot_eigenvalues(net: nn.Module, ID, training_config: Union[str, int]='main', epoch=None)->None:
    """
    plot eigenvalues of RNN weight matrix in complex space

    :param net
    :param epoch: epoch
    :param training_config: int defining configuration or String 'main'
    """
    rnn_weights = net.get_weights()["J"].cpu().detach().numpy()
    if rnn_weights.shape[0] == rnn_weights.shape[1]:
        eigenvalues, _ = np.linalg.eig(rnn_weights)
        fig = plt.figure()
        plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), color='#959798')
        plt.xlabel('Re(Eigenvalues)')
        plt.ylabel('Im(Eigenvalues)')
        plt.box(False)
        if epoch is None:
            plt.savefig(f"{PATH}/{ID}/Main-training/Results/Train/RNN_eigenvalues_distribution.png", transparent=True)
        else:
            plt.savefig(f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{' ' + training_config if training_config!='main' else ''}/RNN_eigenvalues_distribution/{epoch}.png", transparent=True)
        plt.close(fig)


def plot_loss_trajectory(training_config: Union[str, int], ID)->None:
    """
    plot loss trajectory of the network

    :param training_config: int defining configuration or String 'main'
    """

    training_losses = cPickle.load(open(f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{(' ' + str(training_config)) if training_config!='main' else ''}/training_loss.pkl", "rb" ))
    fig = plt.figure()
    plt.plot(np.arange(len(training_losses)), training_losses, 'o-', label='train')
    if training_config=='main':
        testing_losses = cPickle.load(open(f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{(' ' + str(training_config)) if training_config!='main' else ''}/testing_loss.pkl", "rb" ))
        plt.plot(np.arange(len(testing_losses)), testing_losses, 'x-', color='#fa525b', label='test')
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.legend(framealpha=0)
    plt.box(False)
    plt.savefig(f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{(' ' + str(training_config)) if training_config!='main' else ''}/Loss.png", transparent=True)
    plt.close(fig)
