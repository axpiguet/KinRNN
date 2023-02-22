from typing import List
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math

from rnn import BATCH_SIZE, GPU, LR, WORTH_MP
import data

from .interpret import display_net_state, plot, plot_loss_trajectory
from .save import load_data, load_checkpoint, dump_checkpoint, dump_loss


def regul_L1(net: nn.Module)->float:
    """
    Encourage sparsity of weights
    """
    RL1 = torch.sum(torch.abs(net.get_weights()["J"])) + torch.sum(torch.abs(net.get_weights()["W"])) + torch.sum(torch.abs(net.get_weights()["B"]))
    return RL1


def regul_L2(net: nn.Module)->float:
    """
    Homogeanize weights
    """
    RL2 = torch.sum(torch.square(net.get_weights()["B"])) + torch.sum(torch.square(net.get_weights()["J"])) + torch.sum(torch.square(net.get_weights()["W"]))
    return RL2


def regul_FR(activity: torch.Tensor)->float:
    """
    Encourage sparsity of firing rates
    """
    RFR = torch.mean(torch.square(activity))
    return RFR


def train_epoch(net: nn.Module, device: torch.device, dataloader: DataLoader, criterion, optimizer, beta1: float=None, beta2: float=None, beta_FR: float=None,kin = False)->float:
    net.train()
    batch_losses = []
    for (input_batch, label_batch) in dataloader:

        input_batch = torch.fliplr(torch.rot90(input_batch, k=-1)).to(device, non_blocking=True)
        label_batch = torch.fliplr(torch.rot90(label_batch, k=-1)).to(device, non_blocking=True)#.float()
        net.show_grad()
        optimizer.zero_grad()
        output, activity = net(input_batch)
        if kin :
            print('indeed')
            output = torch.cat((output, activity), 2)
        print(output)
        lossang  = 1*criterion(output[:,:,0:2], label_batch[:,:,0:2])
        lossemg  = criterion(output[:,:,2:9], label_batch[:,:,2:9])
        alpha =0.1#0.1#1
        beta =1#1
        #if (lossemg>3*lossang  ):
        #    alpha = 0.1
        loss = alpha*criterion(output[:,:,0:2],label_batch[:,:,0:2])+ beta*criterion(output[:,:,2:9], label_batch[:,:,2:9])#criterion(output, label_batch)

        #loss =  torch.mean(torch.sum((output[:,:,2:9]- label_batch[:,:,2:9])**2))/4#beta*criterion(torch.permute(output[:,:,2:9], (1, 2, 0)), torch.permute(label_batch[:,:,2:9], (1, 2, 0)))

        print('angle loss : ' , int(lossang.clone().item()))
        print('emg loss   : ' ,int(lossemg.clone().item()))
        print('    loss   : ',int(loss.clone().item()))
        fig, ax = plt.subplots(1, 1)
        plt.plot(np.linspace(0,1000,label_batch.shape[0]), label_batch.cpu()[:,0,2:9],'b')
        plt.plot(np.linspace(0,1000,label_batch.shape[0]), output.clone().detach().cpu()[:,0,2:9],'r')
        plt.savefig("emg.png")

        if beta1 != None:
            loss += beta1 * regul_L1(net)
        if beta2 != None:
            loss += beta2 * regul_L2(net)
        if beta_FR != None:
            loss+= beta_FR * regul_FR(activity)

        #with torch.autograd.profiler.profile() as prof:
        loss.backward()
        #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        clipping_value = 1 # arbitrary value of your choosing
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1200)

        optimizer.step()    # Does the update
        #print(net.get_weightsgrad())
        batch_losses.append(loss.item())

    running_loss = np.mean(batch_losses)
    return running_loss


def test_epoch(net: nn.Module, device: torch.device, dataloader: DataLoader, criterion, beta1: float=None, beta2: float=None, beta_FR: float=None , plot = False, kin = False)->float:
    net.eval()
    batch_losses = []
    batch_i = 0
    count = 0
    for (input_batch, label_batch) in dataloader:
        input_batch = torch.fliplr(torch.rot90(input_batch, k=-1)).to(device, non_blocking=True)
        label_batch = torch.fliplr(torch.rot90(label_batch, k=-1)).to(device, non_blocking=True).float()


        #print('TEST       ' , np.shape(input_batch))
        output, activity = net(input_batch)


        with open('input.npy', 'wb') as f:
            np.save(f, input_batch.clone().detach().numpy(),allow_pickle=False)

        with open('truth.npy', 'wb') as f:
            np.save(f, label_batch.clone().detach().numpy(), allow_pickle=False)


        with open('pred.npy', 'wb') as f:
            np.save(f,torch.cat((output, activity), 2).clone().detach().numpy(), allow_pickle=False)


        if kin :
            output = torch.cat((output, activity), 2)# anlges then emg

        loss = criterion(output[:,:,0:9], label_batch[:,:,0:9])
        #loss = criterion(output, label_batch)
        if beta1 != None: loss += beta1 * regul_L1(net)
        if beta2 != None: loss += beta2 * regul_L2(net)
        if beta_FR != None: loss+= beta_FR * regul_FR(activity)

        batch_losses.append(loss.item())
        if plot :
            for image in range(label_batch.shape[1]):
                fig, ax = plt.subplots(1, 1)
                plt.plot(np.linspace(0,1000,label_batch.shape[0]), label_batch.cpu()[:,image,0], label = 'true Hip', color = 'mediumaquamarine', linewidth=3, alpha = 0.3)
                plt.plot(np.linspace(0,1000,label_batch.shape[0]), label_batch.cpu()[:,image,1], label = 'true Knee',color = 'orchid', linewidth=3, alpha = 0.3)
                plt.plot(np.linspace(0,1000,label_batch.shape[0]), output.detach().cpu()[:,image ,0],label = 'pred Hip',color = 'mediumaquamarine' , linewidth=1)
                plt.plot(np.linspace(0,1000,label_batch.shape[0]) , output.detach().cpu()[:,image ,1],label = 'pred Knee', color = 'orchid', linewidth=1)
                plt.ylim([-20,50])
                plt.legend()
                #plt.savefig("test_kin/trial_"+str(count)+".png")
                ax_col = '#585759'
                fig, ax = plt.subplots(1, figsize=(13, 4),frameon = False)
                plt.plot(np.linspace(0,1000,label_batch.shape[0]), output.detach().cpu()[:,image ,1],linewidth=3, color = '#19D3C5', label ='Prediction - Knee')
                plt.plot(np.linspace(0,1000,label_batch.shape[0]), label_batch.cpu()[:,image,1], color = '#19D3C5', linewidth=6, alpha = 0.3, label ='Ground truth - Knee')
                plt.plot(np.linspace(0,1000,label_batch.shape[0]), output.detach().cpu()[:,image ,0],linewidth=3, color = '#FA525B', label ='Prediction - Hip')
                plt.plot(np.linspace(0,1000,label_batch.shape[0]), label_batch.cpu()[:,image,0], color = '#FA525B', linewidth=6, alpha = 0.3, label ='Ground truth - Hip')
                for pos in ['right', 'top']:
                    plt.gca().spines[pos].set_visible(False)
                plt.gca().spines['left'].set_color(ax_col)
                plt.gca().spines['bottom'].set_color(ax_col)
                plt.tick_params(axis='both', colors = ax_col)
                plt.ylabel('Angle [degree]', color = ax_col)
                #plt.title('Knee joint angle', color = ax_col)
                plt.xlabel('Time [ms]',color = ax_col)
                plt.ylim([-100,100])
                plt.legend()
                plt.tight_layout()
                plt.savefig("test_kin/trialreport_"+str(count)+".png", transparent = True )

                muscle_names = ['Il', 'RF', 'VLat', 'ST','TA', 'MG', 'Sol']

                fig, ax = plt.subplots(len(muscle_names), 1, figsize=(17, 13) ,frameon = False)
                for m in range(len(muscle_names)):
                    ax[m].plot(np.linspace(0,1000,label_batch.shape[0]), label_batch.cpu()[:,image,2+m], label = 'Ground truth', color = '#fa525b',  linewidth=6, alpha = 0.15)
                    ax[m].plot(np.linspace(0,1000,label_batch.shape[0]) , activity.detach().cpu()[:,image,m], label = 'Prediction',ls= '--', linewidth=2, color='#fa525b')

                    for pos in ['right', 'top']:
                        ax[m].spines[pos].set_visible(False)
                    ax[m].spines['left'].set_color(ax_col)
                    ax[m].spines['bottom'].set_color(ax_col)
                    ax[m].set_ylabel('Amplitude [u.a.]', color = ax_col)
                    ax[m].set_xlabel(' ',color = ax_col)
                    ax[m].set_ylim([-50,50])
                    ax[m].tick_params(axis='both', colors = ax_col)
                ax[m].set_xlabel('Time [ms]',color = ax_col)
                #plt.plot(np.linspace(0,1000,label_batch.shape[0]) , net.get_inBiomech(activity).detach().cpu()[:,image,:], linewidth=1)
                #ax[0].plot(np.linspace(0,1000,input_batch.shape[0]) , input_batch.cpu()[:,image,:], linewidth=1, color='b')
                #print(torch.mean(torch.abs(input_batch.cpu()[:,image,:])))
                ax[0].legend(loc = 'upper right',frameon = False)
                plt.savefig("test_kin/intermediate_"+str(count)+".png", transparent = True)
                fig, ax = plt.subplots(1, 1)
                plt.plot(np.linspace(0,1000,label_batch.shape[0]) ,net.get_inBiomech(activity).detach().cpu()[:,image,:], linewidth=1)
                plt.legend(["Il", "GM", "RF", "ST", "VLat", 'BF', "MG"])
                plt.savefig("test_kin/activity_"+str(count)+".png")
                count = count + 1
        batch_i = batch_i +1

    running_loss = np.mean(batch_losses)
    return running_loss


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1./np.exp(-0.5*x)

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda

#, input_names: List[str]
def train_emg(net: nn.Module, ID: str, training_sets: torch.Tensor=None, training_targets: torch.Tensor=None,
        n_iterations: int=5000, beta1: float=None, beta2: float=None, beta_FR: float=None,
            config: str='main', perc_reduction: float=0.5, alpha_init=0.2, try_checkpoint: bool=False, multiprocessing: bool=False , BATCHS = 1)->None:
    """
    train net on one or all configurations

    :param net: artificial net
    :param inputs: net tensor inputs of size(data_size, variances, seq_length, input_size)
    :param labels: labels of inputs of size(data_size, seq_length, output_size)
    :param input_names: dictionnary with key name, sub_names, cathode and anodes
    :param reverse_configs: numpy array output of numpy.unique function with return_inverse=True
    :param batch_size: size of one batch
    :param n_iterations: number of iterations
    :param learning_rate: learning rate
    :param beta1: L1-regularization coefficient
    :param beta2: L2-regularization coefficient
    :param beta_FR: firing-rate regularization coefficient
    :param config: int defining configuration or String 'main'
    :param perc_reduction: percentage of reference loss to be reduced before changing inputs
    :param alpha_init: alpha to initialize the network
    :param try_checkpoint: says if we should try to find a checkpoint
    :param multiprocessing: if true says the pre-training is done with multiple cpus only
    """
    dev = GPU if torch.cuda.is_available() and not multiprocessing else "cpu"
    device = torch.device(dev)
    print("Using device ", dev)

    end_lr = 0.01#0.01
    start_lr =0.01#0.01      #1
    lr_find_epochs = n_iterations
    a = 1 # number per batch ??

    if config!='main': net.__init__(net.input_size, net.hidden_size, net.output_size, alpha_init)
    net.to(device)

    print_step = 5
    checkpoint_step = 1
    plot_step = 1000

    # Use Adam optimizer
    criterion = nn.MSELoss()#reduction = 'sum')
    optimizer = optim.Adam(net.parameters(), lr = start_lr)#lr=LR[1] if config=='main' else LR[0])

    lr_lambda = lambda x: math.exp(x * math.log(end_lr / start_lr) / (lr_find_epochs * a))+ 100*math.exp(4*x * math.log(start_lr / end_lr) / (lr_find_epochs * a))

    #lr_lambda = cyclical_lr(5, min_lr=start_lr, max_lr=end_lr)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    training_losses = []
    testing_losses = []
    lr_find_lr = []
    loaded_epoch = 0
    n_step=len(data.WIDTH_VARIANCES_FACTORS)
    step=0

    # try to load data if training has already been done
    if try_checkpoint:
        try :
            loaded_checkpoint = load_checkpoint(config, ID)


            loaded_epoch = loaded_checkpoint['epoch'] + 1
            training_losses = loaded_checkpoint['training_losses']
            testing_losses = loaded_checkpoint['testing_losses']
            ref_loss = loaded_checkpoint['ref_loss']
            step = loaded_checkpoint['input_step']
            net.load_state_dict(loaded_checkpoint['model_state'])
            optimizer.load_state_dict(loaded_checkpoint['optimizer_state'])

            #torch.save(loaded_checkpoint['model_state']['fcpre.weight'] , 'fcpreW.pt')
            #torch.save(loaded_checkpoint['model_state']['fc.weight'] , 'fcW.pt')
            #torch.save(loaded_checkpoint['model_state']['lstm.bias_ih_l0'] , 'biasihl0.pt')
            #torch.save(loaded_checkpoint['model_state']['lstm.bias_hh_l0'] , 'biashhl0.pt')
            #torch.save(loaded_checkpoint['model_state']['lstm.bias_ih_l1'] , 'biasihl1.pt')
            #torch.save(loaded_checkpoint['model_state']['lstm.bias_hh_l1'] , 'biashhl1.pt')
            #torch.save(loaded_checkpoint['model_state']['lstm.weight_ih_l0'] , 'weightihl0.pt')
            #torch.save(loaded_checkpoint['model_state']['lstm.weight_hh_l0'] , 'weighthhl0.pt')
            #torch.save(loaded_checkpoint['model_state']['lstm.weight_ih_l1'] , 'weightihl1.pt')
            #torch.save(loaded_checkpoint['model_state']['lstm.weight_hh_l1'] , 'weighthhl1.pt')
        except:
            print('No checkpoint found')


    torch.manual_seed(42)

    # load data properly
    if config=='main':
        training_sets, training_targets = load_data('train', ID)
        testing_sets, testing_targets = load_data('test', ID)
        #print(training_sets.shape, training_targets.shape)
        test = TensorDataset(testing_sets, testing_targets)
        test_loader = DataLoader(test, batch_size=BATCHS if config=='main' else BATCHS, shuffle=True)
    train = TensorDataset(training_sets, training_targets)
    train_loader = DataLoader(train, batch_size=BATCHS if config=='main' else BATCHS, shuffle=True)

    for epoch in range(loaded_epoch, n_iterations):
        print('------------ Epoch : ', epoch, ' ------------')
        #print('A')
        running_loss = train_epoch(net, device, train_loader, criterion, optimizer, beta1=beta1, beta2=beta2, beta_FR=beta_FR, kin = True)
        #scheduler.step()
        lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
        print(lr_step)
        lr_find_lr.append(lr_step)
        training_losses.append(running_loss)
        #if config=='main':
            #print('B')
            #testing_loss = test_epoch(net, device, test_loader, criterion, beta1=beta1, beta2=beta2, beta_FR=beta_FR)
            #testing_losses.append(testing_loss)
        #print('C')
        if epoch % print_step == 0: display_net_state(optimizer, running_loss, epoch)#, testing_loss if config=='main' else None)
        if epoch % print_step == 0:
            pred, activity = net(torch.fliplr(torch.rot90(testing_sets, k=-1)).to(device, non_blocking=True))
            fig, ax = plt.subplots(1, 1)
            plt.plot(np.linspace(0,1000,testing_targets.shape[1]), testing_targets.cpu()[0,:,0],'x', label = 'true Hip' , color = 'red')
            plt.plot(np.linspace(0,1000,testing_targets.shape[1]), testing_targets.cpu()[0,:,1],'x', label = 'true Knee',  color = 'green')
            plt.plot(np.linspace(0,1000,testing_targets.shape[1]), pred.clone().detach().cpu()[:,0,0],label = 'pred Hip', color = 'red')
            plt.plot(np.linspace(0,1000,testing_targets.shape[1]) , pred.clone().detach().cpu()[:,0,1],label = 'pred Knee',  color = 'green')
            plt.legend()
            plt.savefig("evolution/kin"+str(epoch)+".png")
            testing_loss = test_epoch(net, device, test_loader, criterion, beta1=beta1, beta2=beta2, beta_FR=beta_FR, plot = True, kin = True)

        if epoch==0: ref_loss=running_loss
        # narrow the input if loss decreased enough
        elif epoch>0 and step+1<n_step and running_loss<=ref_loss*(1-perc_reduction):
            #print('D')
            step+=1
            train = TensorDataset(training_sets, training_targets)
            train_loader = DataLoader(train, batch_size=BATCHS if config=='main' else BATCHS, shuffle=True)
            if config=='main':
                #print('E')
                test = TensorDataset(testing_sets, testing_targets)
                test_loader = DataLoader(test, batch_size=BATCHS if config=='main' else BATCHS, shuffle=True)
            ref_loss = running_loss
        #print('F')
        if epoch % checkpoint_step == 0 and epoch>0:
            dump_checkpoint({'epoch': epoch,'model_state': net.state_dict(),'optimizer_state': optimizer.state_dict(),'training_losses': training_losses, 'testing_losses':testing_losses, 'ref_loss': ref_loss, 'input_step': step}, config, ID)
        #if epoch % plot_step == 0 and epoch>0 and config=='main': plot(net, training_sets[:,step,:,:], training_targets, input_names, reverse_configs, device, ID, training_config=config, epoch=epoch)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    #print('G')
    # save loss and model

    ##### comment this line if don't want to save for pretraining
     #net.save_pretrain()

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.linspace(0,1000,len(training_losses)) ,training_losses, linewidth=1)
    #ax[1].plot(np.linspace(0,1000,len(training_losses)) ,lr_find_lr, linewidth=1)
    #ax[0].set_ylim( [0,8])
    plt.savefig("losssss.png")

    dump_checkpoint({'epoch': n_iterations-1,'model_state': net.state_dict(),'optimizer_state': optimizer.state_dict(),'training_losses': training_losses, 'testing_losses':testing_losses, 'ref_loss': ref_loss, 'input_step': step}, config, ID)
    dump_loss(training_losses, 'train', config, ID)
    dump_loss(testing_losses, 'test', config, ID)



def train(net: nn.Module, input_names: List[str], reverse_configs: np.ndarray, ID: str, training_sets: torch.Tensor=None, training_targets: torch.Tensor=None,
        n_iterations: int=5000, beta1: float=None, beta2: float=None, beta_FR: float=None,
            config: str='main', perc_reduction: float=0.5, alpha_init=0.2, try_checkpoint: bool=False, multiprocessing: bool=False)->None:
    """
    train net on one or all configurations

    :param net: artificial net
    :param inputs: net tensor inputs of size(data_size, variances, seq_length, input_size)
    :param labels: labels of inputs of size(data_size, seq_length, output_size)
    :param input_names: dictionnary with key name, sub_names, cathode and anodes
    :param reverse_configs: numpy array output of numpy.unique function with return_inverse=True
    :param batch_size: size of one batch
    :param n_iterations: number of iterations
    :param learning_rate: learning rate
    :param beta1: L1-regularization coefficient
    :param beta2: L2-regularization coefficient
    :param beta_FR: firing-rate regularization coefficient
    :param config: int defining configuration or String 'main'
    :param perc_reduction: percentage of reference loss to be reduced before changing inputs
    :param alpha_init: alpha to initialize the network
    :param try_checkpoint: says if we should try to find a checkpoint
    :param multiprocessing: if true says the pre-training is done with multiple cpus only
    """
    dev = GPU if torch.cuda.is_available() and not multiprocessing else "cpu"
    device = torch.device(dev)
    print("Using device ", dev)

    if config!='main': net.__init__(net.input_size, net.hidden_size, net.output_size, alpha_init)
    net.to(device)

    print_step = 50
    checkpoint_step = 100
    plot_step = 1000

    # Use Adam optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR[1] if config=='main' else LR[0])

    training_losses = []
    testing_losses = []
    loaded_epoch = 0
    n_step=len(data.WIDTH_VARIANCES_FACTORS)
    step=0

    # try to load data if training has already been done
    if try_checkpoint:
        try :
            loaded_checkpoint = load_checkpoint(config, ID)
            loaded_epoch = loaded_checkpoint['epoch'] + 1
            training_losses = loaded_checkpoint['training_losses']
            testing_losses = loaded_checkpoint['testing_losses']
            ref_loss = loaded_checkpoint['ref_loss']
            step = loaded_checkpoint['input_step']
            net.load_state_dict(loaded_checkpoint['model_state'])
            optimizer.load_state_dict(loaded_checkpoint['optimizer_state'])
        except:
            print('No checkpoint found')

    torch.manual_seed(42)

    # load data properly
    if config=='main':
        training_sets, training_targets = load_data('train', ID)
        testing_sets, testing_targets = load_data('test', ID)
        #print(training_sets.shape, training_targets.shape)
        test = TensorDataset(testing_sets[:,step,:,:], testing_targets)
        test_loader = DataLoader(test, batch_size=BATCH_SIZE[1] if config=='main' else BATCH_SIZE[0], shuffle=True)
    train = TensorDataset(training_sets[:,step,:,:], training_targets)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE[1] if config=='main' else BATCH_SIZE[0], shuffle=True)
    print('train loader ' , train_loader)

    for epoch in range(loaded_epoch, n_iterations):
        running_loss = train_epoch(net, device, train_loader, criterion, optimizer, beta1=beta1, beta2=beta2, beta_FR=beta_FR)
        training_losses.append(running_loss)
        if config=='main':
            testing_loss = test_epoch(net, device, test_loader, criterion, beta1=beta1, beta2=beta2, beta_FR=beta_FR)
            testing_losses.append(testing_loss)

        if epoch % print_step == 0: display_net_state(optimizer, running_loss, epoch, testing_loss if config=='main' else None)
        if epoch==0: ref_loss=running_loss
        # narrow the input if loss decreased enough
        elif epoch>0 and step+1<n_step and running_loss<=ref_loss*(1-perc_reduction):
            step+=1
            train = TensorDataset(training_sets[:,step,:,:], training_targets)
            train_loader = DataLoader(train, batch_size=BATCH_SIZE[1] if config=='main' else BATCH_SIZE[0], shuffle=True)
            if config=='main':
                test = TensorDataset(testing_sets[:,step,:,:], testing_targets)
                test_loader = DataLoader(test, batch_size=BATCH_SIZE[1] if config=='main' else BATCH_SIZE[0], shuffle=True)
            ref_loss = running_loss
        if epoch % checkpoint_step == 0 and epoch>0:
            dump_checkpoint({'epoch': epoch,'model_state': net.state_dict(),'optimizer_state': optimizer.state_dict(),'training_losses': training_losses, 'testing_losses':testing_losses, 'ref_loss': ref_loss, 'input_step': step}, config, ID)
        #if epoch % plot_step == 0 and epoch>0 and config=='main': plot(net, training_sets[:,step,:,:], training_targets, input_names, reverse_configs, device, ID, training_config=config, epoch=epoch)
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # save loss and model
    dump_checkpoint({'epoch': n_iterations-1,'model_state': net.state_dict(),'optimizer_state': optimizer.state_dict(),'training_losses': training_losses, 'testing_losses':testing_losses, 'ref_loss': ref_loss, 'input_step': step}, config, ID)
    dump_loss(training_losses, 'train', config, ID)
    dump_loss(testing_losses, 'test', config, ID)
    #plot(net, training_sets[:,step,:,:], training_targets, input_names, reverse_configs, device, ID, split='Train')
    plot_loss_trajectory(config, ID)


def train_per_configs(net: nn.Module, inputs: np.ndarray, labels: np.ndarray, input_names: List[str], reverse_configs: np.ndarray, method: str, alpha_init: List[float],
        ID, n_iterations: List[int]=[1000,3000], beta1: List[float]=[None,None], beta2: List[float]=[None,None], beta_FR: List[float]=[None,None], perc_reduction: float=0.5)->None:
    """
    train net per configuration, assemble net parameters with method and finally train the net on all configurations

    :param net: artificial net
    :param inputs: net tensor inputs of size(variances, seq_length, data_size, input_size)
    :param labels: labels of inputs of size(seq_length, data_size, output_size)
    :param input_names: dictionnary with key name, sub_names, cathode and anodes
    :param reverse_configs: numpy array output of numpy.unique function with return_inverse=True
    :param method: reduction method of parameters net
    :param alpha_init: alpha parameters that corresponds
    :param batch_size: size of one batch
    :param n_iterations: number of iterations
    :param learning_rate: learning rate
    :param beta1: L1-regularization coefficient
    :param beta2: L2-regularization coefficient
    :param beta_FR: firing-rate regularization coefficient
    :param config: int defining configuration or String 'main'
    :param perc_reduction: percentage of reference loss to be reduced before changing inputs
    """
    # pre-train the network for each configuration (not the 0 configuration) and save trained parameters
    # multiprocessing of sub-trainings
    print('Bonjour')
    if len(input_names) >= WORTH_MP:
        processes = []
        print(f"Multiprocessing pre-trainings...")
        for config in range(len(input_names)):
            input_config = inputs[np.asarray(reverse_configs==config),:,:,:]
            label_config =  labels[np.asarray(reverse_configs==config),:,:]
            input_config, label_config = data.normalize(input_config.astype('float64')), data.normalize(label_config)
            input_config, label_config = torch.from_numpy(input_config.astype('float32')), torch.from_numpy(label_config)
            p = mp.Process(target=train, args=(net, input_names.iloc[[config],:], np.delete(np.where(reverse_configs==config,0,1),np.asarray(reverse_configs!=config)), input_config,label_config, ID, n_iterations[0], beta1[0], beta2[0], beta_FR[0], input_names['name'][config], perc_reduction, alpha_init[np.asarray(reverse_configs==config)], True,True,))
            p.start()
            processes.append(p)

        for p in processes: p.join()

    # single process sub-trainings
    else:
        for config in range(len(input_names)):
            input_config = inputs[np.asarray(reverse_configs==config),:,:,:]
            label_config =  labels[np.asarray(reverse_configs==config),:,:]
            input_config, label_config = data.normalize(input_config.astype('float64')), data.normalize(label_config)
            input_config, label_config = torch.from_numpy(input_config.astype('float32')), torch.from_numpy(label_config)
            print(f"Pre-training configuration {input_names['name'][config]}")
            train(net, input_names.iloc[[config],:], np.delete(np.where(reverse_configs==config,0,1),np.asarray(reverse_configs!=config)), input_config,label_config, ID, n_iterations[0], beta1[0], beta2[0], beta_FR[0], input_names['name'][config], perc_reduction, alpha_init[np.asarray(reverse_configs==config)], True, False)

    # loads all trained parameters in a dict
    params_config = {}
    for name in net.state_dict().keys(): params_config[name] = []
    for name in input_names['name']:
        params_dict  = load_checkpoint(name, ID)['model_state']
        for name in params_dict.keys():
            params_config[name].append(params_dict[name].cpu().detach().numpy())

    print('Main training')
    # initialize parameters thanks to per-config pre-training
    net.__init__(net.input_size, net.hidden_size, net.output_size, alpha_init)
    net.init_weights(params_config, method=method)

    # train the network on the complete set
    train(net, input_names, reverse_configs, ID, n_iterations=n_iterations[1], beta1=beta1[1], beta2=beta2[1], beta_FR=beta_FR[1], config='main', perc_reduction=perc_reduction, try_checkpoint=False)
