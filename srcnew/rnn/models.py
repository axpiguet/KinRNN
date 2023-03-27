import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np
import math
from scipy import signal
from scipy.signal import hilbert

from data import FS
from .net_initialization import max_initialization, mean_initialization, rank_initialization
import rnn.biomechanicalPT as m

class GRU(nn.Module):
    """
        Simple GRU recurrent neural network
    """
    def __init__(self, input_size, hidden_size, output_size, alpha, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.gru = nn.GRU(input_size, hidden_size, dtype=torch.float32)
        self.fc = nn.Linear(hidden_size, output_size, dtype=torch.float32)
        self.alpha = torch.zeros(1,hidden_size, dtype=torch.float32)

    def get_weights(self):
        return {"B": self.gru.weight_ih_l0, "J": self.gru.weight_hh_l0, "W": self.fc.weight}

    def forward(self, input):
        gru_output, _  = self.gru(input) # (h0, c0) are initialized to zero
        output = self.fc(gru_output)
        return output, gru_output

class LSTM2NoBiomech(nn.Module):
    """
        Simple LSTM recurrent neural network
    """

    def __init__(self, input_size, hidden_size, output_size, hip0 = -10.0,knee0 = -80.0, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hip0_ = hip0
        self.knee0_ = knee0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dtype=torch.float32, dropout=0.5)
        self.fc = nn.Linear( hidden_size, 2, dtype=torch.float32)
        self.alpha = torch.zeros(1,hidden_size, dtype=torch.float32)
        self.biomech =  m.crazyleg(hip0 ,knee0)
    def get_weightsgrad(self):
        return self.lstm.weight_hh_l0.grad
    def get_weight(self):
        return self.lstm.weight_hh_l0

    def get_weights(self):
        return {"B": self.lstm.weight_ih_l0, "J": self.lstm.weight_hh_l0 + self.lstm.weight_hh_l1, "W": self.fc.weight}

    def forward(self, input):
        sig = nn.Sigmoid()
        lstm_output, _  = self.lstm(input) # (h0, c0) are initialized to zero
        output = self.fc(lstm_output)
        return output.detach(), lstm_output.detach()



class LSTM2Biomech(nn.Module):
    """
        Simple LSTM recurrent neural network
    """
    #def __init__(self, input_size, hidden_size, output_size,hip0 = 80,knee0 = 0,dt = 1/1481.48 ,**kwargs):
    def __init__(self, input_size, hidden_size, output_size,hip0 = 80,knee0 = 0,dt = 1/1481.48 , low_pass = 5,**kwargs):
        super().__init__()
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_ = torch.device(dev)
        #torch.autograd.set_detect_anomaly(True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dtype=torch.float32, dropout=0.5)
        self.fcpre = nn.Linear( hidden_size, 7, dtype=torch.float32, bias = False) #8 = GMax 9 = BF
        self.fcpreGMBF = nn.Linear( 7, 2, dtype=torch.float32, bias = False)
        self.fc = torch.nn.parameter.Parameter(data=torch.tensor([[3.9/100, 7.3/100, 5.8/100, 0.5/100, 2.6/100, 0.5/100, 4.01/100]], device = device_, requires_grad=True))
        
        torch.nn.init.xavier_uniform_(self.fcpreGMBF.weight , gain=1)
        torch.nn.init.xavier_uniform_(self.fcpre.weight , gain=200)#500)#1)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=0.1)
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0,gain=0.1)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l1,gain=1)#2
        nn.init.xavier_uniform_(self.lstm.weight_hh_l1,gain=1)
        nn.init.constant_(self.lstm.bias_ih_l0,0.01)
        nn.init.constant_(self.lstm.bias_hh_l0,0.01)
        nn.init.constant_(self.lstm.bias_ih_l1 ,0.01)
        nn.init.constant_(self.lstm.bias_hh_l1 , 0.01)
        pretrain = True
        if pretrain :
            self.fcpre.weight.data = torch.load( 'overfcpreW.pt')
            self.fcpreGMBF.weight.data = torch.load( 'overfcpreGMBFW.pt')
            self.lstm.bias_ih_l0.data = torch.load( 'over40biasihl0.pt')
            self.lstm.bias_hh_l0.data = torch.load('over40biashhl0.pt')
            self.lstm.bias_ih_l1.data = torch.load( 'over40biasihl1.pt')
            self.lstm.bias_hh_l1.data = torch.load( 'over40biashhl1.pt')
            self.lstm.weight_ih_l0.data = torch.load( 'over40weightihl0.pt')
            self.lstm.weight_hh_l0.data = torch.load( 'over40weighthhl0.pt')
            self.lstm.weight_ih_l1.data = torch.load( 'over40weightihl1.pt')
            self.lstm.weight_hh_l1.data = torch.load( 'over40weighthhl1.pt')

        self.biomech =  m.crazyleg2(hip0 ,knee0 , dt_ = dt)
        self.theta0hip =  self.biomech.get_theta0()[0][0]
        self.theta0knee =  self.biomech.get_theta0()[1][0]

        #filter emg
        low_pass = low_pass
        sfreq = 1/dt
        low_pass = low_pass/(sfreq/2)
        self.b2, self.a2 = signal.butter(4, low_pass, btype='lowpass')

    def get_weightsgrad(self):
        return self.lstm.weight_hh_l0.grad
    def get_weight(self):
        return self.lstm.weight_hh_l0

    def save_pretrain(self):
        torch.save(self.fcpre.weight , 'over40fcpreW.pt')
        torch.save(self.fcpreGMBF.weight , 'over40fcpreGMBFW.pt')
        torch.save(self.lstm.bias_ih_l0 , 'over40biasihl0.pt')
        torch.save(self.lstm.bias_hh_l0 , 'over40biashhl0.pt')
        torch.save(self.lstm.bias_ih_l1 , 'over40biasihl1.pt')
        torch.save(self.lstm.bias_hh_l1 , 'over40biashhl1.pt')
        torch.save(self.lstm.weight_ih_l0 , 'over40weightihl0.pt')
        torch.save(self.lstm.weight_hh_l0 , 'over40weighthhl0.pt')
        torch.save(self.lstm.weight_ih_l1 , 'over40weightihl1.pt')
        torch.save(self.lstm.weight_hh_l1 , 'over40weighthhl1.pt')

    def get_weights(self):
        return {"B": self.lstm.weight_ih_l0, "J": self.lstm.weight_hh_l0 + self.lstm.weight_hh_l1, "W": self.fcpre.weight}

    def get_inBiomech(self,out):
        out_  = torch.from_numpy(signal.filtfilt(self.b2, self.a2, torch.abs(self.sigmoid_(out.clone().detach())).numpy() ,axis = 0).copy())
        out2 = self.fcpreGMBF(out_.float())
        matched_ = torch.zeros(np.shape(out_[:,:,0:7]))
        matched_[:,:,0] = out_[:,:,0]
        matched_[:,:,2] = out_[:,:,1]
        matched_[:,:,3] = out_[:,:,3]
        matched_[:,:,4] = out_[:,:,2]
        matched_[:,:,6] = out_[:,:,5]
        matched_[:,:,1] = 0.5*out2.clone()[:,:,0]  #Gmax
        matched_[:,:,5] =  out2.cl
        matched_out_ = matched_.clone()
        output = torch.mul(matched_out_ , self.fc)
        return torch.abs(output)

    def sigmoid_(self, x):
        return -1 + 2.0 / (1.0 + torch.exp(-(x)/5))

    def show_grad(self):
        if not(self.fcpre.weight.grad == None):
            print('Scaling factors  : ',self.fc)
            print('biomech  ' ,' grad : ', self.fc.grad,' val : ', torch.mean(torch.abs(self.fc)))
            #print('biomech  ' ,' grad : ', torch.mean(torch.abs(self.fc.weight.grad)),' val : ', torch.mean(torch.abs(self.fc.weight)))

            print('PRE  ' ,' grad : ', torch.mean(torch.abs(self.fcpre.weight.grad)),' val : ', torch.mean(torch.abs(self.fcpre.weight)))

            print('IHL0  ' ,torch.mean(torch.abs(self.lstm.bias_ih_l0.grad)),' val : ', torch.mean(torch.abs(self.lstm.bias_ih_l0)))

            print('HHL0  ' ,torch.mean(torch.abs(self.lstm.bias_hh_l0.grad)),' val : ', torch.mean(torch.abs(self.lstm.bias_hh_l0)))

            print('IHL1  ' ,torch.mean(torch.abs(self.lstm.bias_ih_l1.grad)),' val : ', torch.mean(torch.abs(self.lstm.bias_ih_l1)))

            print('HHL1  ' ,torch.mean(torch.abs(self.lstm.bias_hh_l1.grad)),' val : ', torch.mean(torch.abs(self.lstm.bias_hh_l1)))


    def forward(self, input):
        sig = nn.Sigmoid()
        lstm_output, _  = self.lstm(input) # (h0, c0) are initialized to zero

        out = self.fcpre(lstm_output.clone())#.detach())
        out_=out.clone()
        out2 = self.fcpreGMBF(out_.float())
        matched_ = torch.zeros(np.shape(out_[:,:,0:7]))
        matched_[:,:,0] = out_[:,:,0]
        matched_[:,:,2] = out_[:,:,1]
        matched_[:,:,3] = out_[:,:,3]
        matched_[:,:,4] = out_[:,:,2]
        matched_[:,:,5] = out_[:,:,3]
        matched_[:,:,6] = out_[:,:,5]
        matched_[:,:,1] =  out2.clone()[:,:,0]  #Gmax
        matched_[:,:,5] =  out2.clone()[:,:,1] #BF
       
        matched_out_ = matched_.clone()
        output = torch.mul(torch.abs(matched_out_ ), torch.abs(self.fc))
    
###########
      
        angles, activations = self.biomech(torch.abs(torch.permute(output, (2,1,0)))) #let's check if output is of size 6x 1

#############
        self.readout = angles.clone()
        ang = torch.permute(self.readout, (2,1,0))
        ang[:,:,1] = ang.clone()[:,:,1] - ang.clone()[:,:,0] - self.theta0knee
        ang[:,:,0] = ang.clone()[:,:,0] - math.pi/2 - self.theta0hip
        angles_ = torch.rad2deg(ang)
        return angles_ , self.fcpre(lstm_output)


class LSTM2(nn.Module):
    """
        Simple LSTM recurrent neural network
    """

    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dtype=torch.float32, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size, dtype=torch.float32)
        self.alpha = torch.zeros(1,hidden_size, dtype=torch.float32)

    def get_weightsgrad(self):
        return self.lstm.weight_hh_l0.grad

    def get_weights(self):
        return {"B": self.lstm.weight_ih_l0, "J": self.lstm.weight_hh_l0 + self.lstm.weight_hh_l1, "W": self.fc.weight}

    def forward(self, input):
        lstm_output, _  = self.lstm(input) # (h0, c0) are initialized to zero
        output = self.fc(lstm_output)
        return output, lstm_output


class LSTM(nn.Module):
    """
        Simple LSTM recurrent neural network
    """

    def __init__(self, input_size, hidden_size, output_size, alpha, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, dtype=torch.float32)
        self.fc = nn.Linear(hidden_size, output_size, dtype=torch.float32)
        self.alpha = torch.zeros(1,hidden_size, dtype=torch.float32)

    def get_weights(self):
        return {"B": self.lstm.weight_ih_l0, "J": self.lstm.weight_hh_l0, "W": self.fc.weight}

    def forward(self, input):
        lstm_output, _  = self.lstm(input) # (h0, c0) are initialized to zero
        output = self.fc(lstm_output)
        return output, lstm_output


class LSTMsep(nn.Module):
    """
        Independent LSTM recurrent neural networks for each muscle
    """

    def __init__(self, input_size, hidden_size, output_size, alpha, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = int(hidden_size/output_size)
        self.output_size = output_size # number of muscles
        #################
        # index of the liner correspond to the regular muscle order
        self.lstms = nn.ModuleList([nn.LSTM(input_size, self.hidden_size, dtype=torch.float32)for i in range(output_size)])
        self.linears = nn.ModuleList([nn.Linear(self.hidden_size, 1, dtype=torch.float32) for i in range(output_size)])
        ##################
        self.alpha = torch.zeros(1,hidden_size, dtype=torch.float32)

    def get_weights(self):
        for j in range(self.output_size):
            if j == 0 :
                B_ = self.lstms[j].weight_ih_l0
                J_ = self.lstms[j].weight_hh_l0
                W_ = self.linears[j].weight
            else :
                B_ = torch.cat((B_ ,self.lstms[j].weight_ih_l0),0)
                J_ = torch.cat((J_ ,self.lstms[j].weight_hh_l0),0)
                W_ = torch.cat((W_ ,self.linears[j].weight),0)

        return {"B": B_, "J": J_, "W": W_}

    def get_weightsIndiv(self):
        direct = {"B_Il": self.lstms[0].weight_ih_l0, "J_Il": self.lstms[0].weight_hh_l0, "W": self.linears[0].weight}
        for k in range(1,self.output_size):
            direct.update({"B_Il": self.lstms[k].weight_ih_l0, "J_Il": self.lstms[k].weight_hh_l0, "W": self.linears[k].weight})
        return direct


    def forward(self, input):
        for i in range(self.output_size):
            x1 , _ = self.lstms[i](input)
            x = self.linears[i](x1)
            if i == 0 :
                out = x
                lstm_output = x1
            else :
                out = torch.cat((out,x), 2)
                lstm_output = lstm_output+x1
        return out, lstm_output


class FeedforwardNeuralNetModel(nn.Module):
    """
        Independent LSTM recurrent neural networks for each muscle
    """

    def __init__(self, input_size, hidden_size, output_size, alpha, **kwargs):

        self.linears = nn.ModuleList([nn.Linear(hidden_size, 1, dtype=torch.float32) for i in range(output_size)])
        #################
        super(FeedforwardNeuralNetModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = int(hidden_size/output_size)
        self.output_size = output_size # number of muscles
        # Linear function
        self.fc1s = nn.ModuleList([nn.Linear(input_size, hidden_size, dtype=torch.float32)for i in range(output_size)])
        # Non-linearity
        self.sigmoid = nn.Sigmoid()  # do we need several instance of it ?
        # Linear function (readout)
        self.fc2s = nn.ModuleList([nn.Linear(hidden_size, output_size, dtype=torch.float32)for i in range(output_size)])

    def forward(self, x):
        x1 , _ = self.fc1s[0](input)    #can we have doube output ?
        x = self.fc2s[0](self.sigmoid(x1))
        out = x
        activity = x1 # maybe ask marion about that
        for i in range(1,self.output_size):
            x1 , _ = self.fc1s[i](input)
            x = self.fc2s[i](self.sigmoid(x1))

            out = torch.cat((out,x), 2)
            activity = activity+x1
        return out , activity





class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc = nn.Linear(hidden_size, output_size, dtype=torch.float32)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size, dtype=torch.float32)

    def init_weights(self, parameters_dict, method='mean'):
        take_alpha=True
        if method=='mean':
            new_params = mean_initialization(parameters_dict)
        elif method=='max':
            new_params = max_initialization(parameters_dict, take_alpha=take_alpha)
        elif method=='rank':
            new_params = rank_initialization(parameters_dict, self.alpha, take_alpha=take_alpha)
        else:
            raise ValueError(f'"{method}" is not a valid reduction method. Choose amongst "max", "mean" or "rank".')

        for string_name, param in new_params.items():
            split_string_name = string_name.split('.')
            if len(split_string_name) > 1:
                setattr(eval(f'self.{split_string_name[0]}'), split_string_name[1], nn.Parameter(torch.tensor(param)))
            else:
                setattr(self, string_name, nn.Parameter(torch.tensor(param)))

class RNN(NN):
    """Recurrent network model.

    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        alpha: float

    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)
    """
    def __init__(self, input_size, hidden_size, output_size, alpha):
        super().__init__(input_size, hidden_size, output_size)
        alpha = 1
        self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=False)

        #self.input2h = nn.Linear(input_size, hidden_size, dtype=torch.float32)
        self.rnn = nn.RNNCell(input_size, hidden_size, dtype=torch.float32)

    def get_weights(self):
        return {"B": self.rnn.weight_ih, "J": self.rnn.weight_hh, "W": self.fc.weight}

    def recurrence(self, input, hidden):
        # print(input.shape)
        # print(hidden.shape)
        h_new = self.rnn(input, hidden)
        h_new = self.alpha*(h_new - hidden) + hidden
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""

        # If hidden activity is not provided, initialize it
        if hidden is None: hidden = self.init_hidden(input.shape).to(input.device)

        # Loop through time
        rnn_output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden).to(input.device)
            rnn_output.append(hidden)

        # Stack together output from all time steps
        rnn_output = torch.stack(rnn_output, dim=0)  # (seq_len, batch, hidden_size)
        out = self.fc(rnn_output)
        return out, rnn_output



class ffEIRNN(RNN):
    """E-I RNN.

    Reference:
        Song, H.F., Yang, G.R. and Wang, X.J., 2016.
        Training excitatory-inhibitory recurrent neural networks
        for cognitive tasks: a simple and flexible framework.
        PLoS computational biology, 12(2).

    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons

    Inputs:
        input: (seq_len, batch, input_size)
        hidden: (batch, hidden_size)
        e_prop: float between 0 and 1, proportion of excitatory neurons
    """

    def __init__(self, input_size, hidden_size, output_size, alpha=0.2, alphasimple=0.6,
                 sigma_rec=1, **kwargs):
        super().__init__(input_size, hidden_size, output_size, alpha=alpha)
        self.num_layers = 1
        self.hidden_size = int(hidden_size/ output_size)
        print('hidden size is : ', self.hidden_size)
        self.output_size = output_size
        # Recurrent noise
        #alpha_ = alphasimple * torch.ones(self.output_size)
        #self.alpha_ = alphasimple * torch.ones(self.hidden_size,self.output_size)
        self.alpha_= torch.normal(0.5, 1, size=(self.hidden_size,self.output_size))
        self.alpha_ = (self.alpha_ -torch.min(self.alpha_, 0).values)/(torch.max(self.alpha_, 0).values-torch.min(self.alpha_, 0).values)
        print(self.alpha_)
        self._sigma_rec = nn.Parameter(torch.sqrt(2*self.alpha_) * sigma_rec, requires_grad=True)
        #self._sigma_rec = torch.sqrt(2*alpha_) * sigma_rec
        #self.input2h = nn.Linear(input_size, hidden_size, dtype=torch.float32)
        self.input2hs = nn.ModuleList([nn.Linear(input_size, self.hidden_size, dtype=torch.float32) for i in range(output_size)])
        #self.h2h = EIRecLinear(hidden_size, e_prop=0.8)
        #self.fc = nn.Linear(hidden_size, output_size, dtype=torch.float32)
        self.fcs = nn.ModuleList([ nn.Linear(self.hidden_size, 1, dtype=torch.float32) for i in range(output_size)])

    def init_hidden(self, input):
        batch_size = input.shape[1]
        return (torch.zeros(batch_size, self.hidden_size,self.output_size, requires_grad=True).to(input.device),torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(input.device))

    def get_weightsIndiv(self):
        direct = {"B_Il": self.input2hs[0].weight, "J_Il": torch.tensor([[0]]), "W": self.fcs[0].weight}
        for k in range(1,self.output_size):
            direct.update({"B_Il": self.input2hs[k].weight, "J_Il": torch.tensor([[0]]), "W": self.fcs[k].weight})
        return direct

    def get_alphas(self):
        return self._sigma_rec

    def get_weights(self):
        B_ = self.input2hs[0].weight
        W_ = self.fcs[0].weight
        for j in range(1,self.output_size):

            B_ = torch.cat((B_ ,self.input2hs[j].weight),0)
            W_ = torch.cat((W_ ,self.fcs[j].weight),0)

        return {"B": B_, "J": torch.tensor([[0]]), "W": W_}

    def recurrence(self, input, hidden):
        """Recurrence helper."""
        state, output = hidden
        ##
        x1 = self.input2hs[0](input)    #can we have doube output ?
        total_input = torch.unsqueeze(x1,2) # maybe ask marion about that

        for i in range(1,self.output_size):

            x1 = self.input2hs[i](input)

            #total_input = torch.cat((total_input,torch.unsqueeze(x1,0)), 0)
            total_input = torch.cat((total_input,torch.unsqueeze(x1,2)), 2)
        ##
        state = self.alpha.to(torch.float32)*(total_input - state) + state
        state += self._sigma_rec * torch.randn_like(state, device=input.device)
        output = torch.tanh(state)

        return state, output

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input)

        self._sigma_rec.to(input.device)
        rnn_activity = []
        steps = range(input.size(0))
        for j in steps:
            hidden = self.recurrence(input[j], hidden)
            rnn_activity.append(hidden[1])# consider the dim
        rnn_activity = torch.stack(rnn_activity, dim=0)

        out = self.fcs[0](rnn_activity[:,:,:,0])
        for k in range(1,self.output_size):
            ## change slice
            x = self.fcs[k](rnn_activity[:,:,:,k])
            out = torch.cat((out,x), 2)
        return out, rnn_activity


class FffEIRNN(RNN):

    def __init__(self, input_size, hidden_size, output_size, alpha=0.2, alphasimple=0.6,
                 sigma_rec=1, **kwargs):
        super().__init__(input_size, hidden_size, output_size, alpha=alpha)
        self.num_layers = 1
        self.hidden_size = int(hidden_size/output_size)
        print('hidden size ', self.hidden_size)
        self.output_size = output_size
        self.feedback_delay = int(50*FS/1000)
        #alpha_ = alphasimple * torch.ones(self.output_size)
        #self.alpha_ = alphasimple * torch.ones(self.hidden_size,self.output_size)
        self.alpha_= torch.normal(0.5, 1, size=(self.hidden_size,self.output_size))
        self.alpha_ = (self.alpha_ -torch.min(self.alpha_, 0).values)/(torch.max(self.alpha_, 0).values-torch.min(self.alpha_, 0).values)

        self._sigma_rec = nn.Parameter(torch.sqrt(2*self.alpha_) * sigma_rec, requires_grad=True)
        #self._sigma_rec = torch.sqrt(2*alpha_) * sigma_rec
        #self.input2h = nn.Linear(input_size, hidden_size, dtype=torch.float32)
        self.input2hs = nn.ModuleList([nn.Linear(input_size, self.hidden_size, dtype=torch.float32) for i in range(output_size)])
        #self.h2h = EIRecLinear(hidden_size, e_prop=0.8)
        #self.fc = nn.Linear(hidden_size, output_size, dtype=torch.float32)
        self.fcs = nn.ModuleList([ nn.Linear(self.hidden_size, 1, dtype=torch.float32) for i in range(output_size)])
        #self.f2h = nn.Linear(output_size , hidden_size , dtype = torch.float32)
        self.f2hs = nn.ModuleList([ nn.Linear(output_size , self.hidden_size, dtype=torch.float32) for i in range(output_size)])

    def init_feedback(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.output_size, dtype=torch.float32)

    def init_hidden(self, input):
        batch_size = input.shape[1]
        return (torch.zeros(batch_size, self.hidden_size,self.output_size, requires_grad=True).to(input.device),torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(input.device))

    def get_weightsIndiv(self):
        direct = {"B_Il": self.input2hs[0].weight, "J_Il": torch.tensor([[0]]), "W": self.fcs[0].weight}
        for k in range(1,self.output_size):
            direct.update({"B_Il": self.input2hs[k].weight, "J_Il": torch.tensor([[0]]), "W": self.fcs[k].weight})
        return direct

    def get_alphas(self):
        return self._sigma_rec

    def get_weights(self):
        B_ = self.input2hs[0].weight
        W_ = self.fcs[0].weight
        for j in range(1,self.output_size):

            B_ = torch.cat((B_ ,self.input2hs[j].weight),0)
            W_ = torch.cat((W_ ,self.fcs[j].weight),0)

        return {"B": B_, "J": torch.tensor([[0]]), "W": W_}

    def recurrence(self, input, hidden, feedback):
        """Recurrence helper."""
        state, output = hidden
        ##
        x1 = self.input2hs[0](input)    #can we have doube output ?
        total_input = torch.unsqueeze(x1,2) # maybe ask marion about that

        for i in range(1,self.output_size):
            x1 = torch.tanh(self.input2hs[i](input) + self.f2hs[i](feedback))

            #total_input = torch.cat((total_input,torch.unsqueeze(x1,0)), 0)
            total_input = torch.cat((total_input,torch.unsqueeze(x1,2)), 2)
        ##

        state = self.alpha.to(torch.float32)*(total_input - state) + state
        state += self._sigma_rec * torch.randn_like(state, device=input.device)
        output = torch.tanh(state)

        return state, output

    def forward(self, input, hidden=None , feedback = None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input)
        if feedback is None:
            feedback = self.init_feedback(input.shape).to(input.device)

        self._sigma_rec.to(input.device)
        rnn_activity = []
        steps = range(input.size(0))
        readout =  []
        for j in steps:
            if j < self.feedback_delay:
                hidden = self.recurrence(input[j], hidden, feedback)
            else:
                hidden = self.recurrence(input[j], hidden, readout[j-self.feedback_delay])

            #hidden = self.recurrence(input[j], hidden, feedback)
            rnn_activity.append(hidden[1])# consider the dim
            #stack.append(hidden[1])
            #rnn_activity = torch.stack(rnn_activity, dim=0)
            out = torch.unsqueeze(self.fcs[0](rnn_activity[j][:,:,0]),2)
            for k in range(1,self.output_size):
                ## change slice
                #x = self.fcs[k](rnn_activity[:,:,:,k])
                x = self.fcs[k](rnn_activity[j][:,:,k])
                x = torch.unsqueeze(x,2)
                out = torch.cat((out,x), 2)
            out = torch.squeeze(out)
            readout.append(out)
        readout = torch.stack(readout, dim=0)
        rnn_activity = torch.stack(rnn_activity, dim=0)

        return readout, rnn_activity



class EIRecLinear(nn.Module):
    """Recurrent E-I Linear transformation.

    Args:
        hidden_size: int, layer size
        e_prop: float between 0 and 1, proportion of excitatory units
    """
    __constants__ = ['bias', 'hidden_size', 'e_prop']

    def __init__(self, hidden_size, e_prop, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.e_prop = e_prop
        self.e_size = int(e_prop * hidden_size)
        self.i_size = hidden_size - self.e_size
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        mask = np.tile([1]*self.e_size+[-1]*self.i_size, (hidden_size, 1))
        np.fill_diagonal(mask, 0)
        self.register_buffer('mask',torch.tensor(mask, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Scale E weight by E-I ratio
        self.weight.data[:, :self.e_size] /= (self.e_size/self.i_size)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def effective_weight(self):
        return torch.abs(self.weight) * self.mask

    def forward(self, input):
        # weight is non-negative
        return F.linear(input, self.effective_weight(), self.bias)


class EIRNN(RNN):
    """E-I RNN.

    Reference:
        Song, H.F., Yang, G.R. and Wang, X.J., 2016.
        Training excitatory-inhibitory recurrent neural networks
        for cognitive tasks: a simple and flexible framework.
        PLoS computational biology, 12(2).

    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons

    Inputs:
        input: (seq_len, batch, input_size)
        hidden: (batch, hidden_size)
        e_prop: float between 0 and 1, proportion of excitatory neurons
    """

    def __init__(self, input_size, hidden_size, output_size, alpha=0.2,
                 e_prop=0.8, sigma_rec=0, **kwargs):
        super().__init__(input_size, hidden_size, output_size, alpha=alpha)
        self.e_size = int(hidden_size * e_prop)
        self.i_size = hidden_size - self.e_size
        self.num_layers = 1
        # Recurrent noise
        self._sigma_rec = np.sqrt(2*alpha) * sigma_rec

        self.input2h = nn.Linear(input_size, hidden_size, dtype=torch.float32)
        self.h2h = EIRecLinear(hidden_size, e_prop=0.8)
        self.fc = nn.Linear(self.e_size, output_size, dtype=torch.float32)

    def init_hidden(self, input):
        batch_size = input.shape[1]
        return (torch.zeros(batch_size, self.hidden_size).to(input.device),
                torch.zeros(batch_size, self.hidden_size).to(input.device))

    def get_weights(self):
        return {"B": self.input2h.weight, "J": self.h2h.effective_weight(), "W": self.fc.weight}

    def recurrence(self, input, hidden):
        """Recurrence helper."""
        state, output = hidden
        total_input = self.input2h(input) + self.h2h(output)
        state = self.alpha.to(torch.float32)*(total_input - state) + state
        state += self._sigma_rec * torch.randn_like(state, device=input.device)
        output = torch.tanh(state)

        return state, output

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input)

        self._sigma_rec.to(input.device)
        rnn_activity = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            print(np.shape(hidden))
            rnn_activity.append(hidden[1])

        rnn_activity = torch.stack(rnn_activity, dim=0)
        rnn_e = rnn_activity[:, :, :self.e_size]
        out = self.fc(rnn_e)
        return out, rnn_activity


############################## ADAPTATIVE TIME CONSTANTS ############################

class ARNN(RNN):
    """
        Reccurent neural network model that derives from adaptative continuous time RNN with adaptative time constants for each neuron

    """

    def __init__(self, input_size, hidden_size, output_size, alpha):

        super().__init__(input_size, hidden_size, output_size, alpha)

        # build a tensor alpha of hidden size
        unique_alphas = np.unique(alpha)
        alphas = torch.randint(0, len(unique_alphas),(1,self.hidden_size), dtype=torch.float64) if len(unique_alphas)!=0 else torch.zeros((1,self.hidden_size), dtype=torch.float64)
        for i_alpha, unique_alpha in enumerate(unique_alphas):
            alphas = torch.where(alphas==i_alpha, unique_alpha, alphas)
        alphas = alphas.to(torch.float32)
        self.alpha = nn.Parameter(alphas, requires_grad=True)


class AEIRNN(EIRNN):
    """
        Reccurent neural network model that derives from adaptative continuous time Excitatory-Inhibitory RNN with adaptative time constants for each neuron

    """

    def __init__(self, input_size, hidden_size, output_size, alpha,
                 e_prop=0.8, sigma_rec=0, **kwargs):
        super().__init__(input_size, hidden_size, output_size, e_prop=e_prop, sigma_rec=sigma_rec)

        # build a tensor alpha of hidden size
        unique_alphas = np.unique(alpha)
        alphas = torch.randint(0, len(unique_alphas),(1,self.hidden_size), dtype=torch.float64)

        for i_alpha, unique_alpha in enumerate(unique_alphas):
            alphas = torch.where(alphas==i_alpha, unique_alpha, alphas)
        self.alpha = nn.Parameter(alphas, requires_grad=True)
        # Recurrent noise
        self.alpha_copy = self.alpha.clone().detach()
        delattr(self, '_sigma_rec')
        self.register_buffer('_sigma_rec', torch.sqrt(2*self.alpha_copy) * sigma_rec)


######################### ADAPTATIVE TIME CONSTANTS WITH SYNAPSE ####################


class ASRNN(NN):
    """
        Reccurent neural network model that derives from adaptative continuous time RNN with additional equation for synaptic transmission
    """

    def __init__(self, input_size, hidden_size, output_size, alpha):
        super().__init__(input_size, hidden_size, output_size)
        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

        unique_alphas = np.unique(alpha)
        alphas = torch.randint(0, len(unique_alphas),(1,self.hidden_size), dtype=torch.float64) if len(unique_alphas)!=0 else torch.zeros((1,self.hidden_size), dtype=torch.float64)
        for i_alpha, unique_alpha in enumerate(unique_alphas):
            alphas = torch.where(alphas==i_alpha, unique_alpha, alphas)
        alphas=torch.sqrt(alphas)
        alphas = alphas.to(torch.float32)
        alpha_s = alphas.detach().clone()

        self.alpha = nn.Parameter(alphas, requires_grad=True)
        self.alpha_s = nn.Parameter(alpha_s, requires_grad=True)

    def get_weights(self):
        return {"B": self.input2h.weight, "J": self.h2h.weight, "W": self.fc.weight}

    def recurrence(self, input, hidden, synapse):
        s_new = self.alpha_s*(self.input2h(input) + self.h2h(hidden) - synapse) + synapse # feedback comes back in the synapse
        h_new = self.alpha*(torch.tanh(s_new) - hidden) + hidden
        return h_new, s_new

    def forward(self, input, hidden=None, synapse=None):
        """Propogate input through the network."""

        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)
        if synapse is None:
            synapse = self.init_hidden(input.shape).to(input.device)

        # Loop through time
        rnn_output = []
        steps = range(input.size(0))
        for i in steps:
            hidden, synapse = self.recurrence(input[i], hidden, synapse)
            hidden.to(input.device)
            synapse.to(input.device)
            rnn_output.append(hidden)

        # Stack together output from all time steps
        rnn_output = torch.stack(rnn_output, dim=0)  # (seq_len, batch, hidden_size)
        out = self.fc(rnn_output)
        return out, rnn_output

class ASEIRNN(AEIRNN):
    """
        Reccurent neural network model that derives from adaptative continuous time Excitatory-Inhibitory RNN with additional feedback
    """

    def __init__(self, input_size, hidden_size, output_size, alpha,
                 e_prop=0.8, sigma_rec=0, **kwargs):
        super().__init__(input_size, hidden_size, output_size, alpha, e_prop=e_prop, sigma_rec=sigma_rec)

        alpha_s = self.alpha.detach().clone()
        self.alpha_s = nn.Parameter(alpha_s, requires_grad=True)

    def get_weights(self):
        return {"B": self.input2h.weight, "J": self.h2h.effective_weight(), "W": self.fc.weight}

    def recurrence(self, input, hidden, synapse):
        """Recurrence helper."""
        state, output = hidden
        state = state.to(torch.float32)
        output = output.to(torch.float32)
        s_new = self.alpha_s*(self.input2h(input) + self.h2h(output) - synapse) + synapse
        state = self.alpha*(torch.tanh(s_new) - state) + state
        state += self._sigma_rec * torch.randn_like(state, device=input.device)
        output = torch.tanh(state)
        return (state, output), s_new


    def forward(self, input, hidden=None, synapse=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input)
        if synapse is None:
            synapse = self.init_hidden(input)[0].to(input.device)

        # Loop through time
        rnn_output = []
        out = []
        steps = range(input.size(0))
        for i in steps:
            hidden, synapse = self.recurrence(input[i], hidden, synapse)
            synapse.to(input.device)
            rnn_output.append(hidden[1])
            rnn_e = hidden[1][:, :self.e_size]
            out.append(self.fc(rnn_e.to(torch.float32)))

        # Stack together output from all time steps
        out = torch.stack(out, dim=0)
        rnn_output = torch.stack(rnn_output, dim=0)
        return out, rnn_output

############################## ADAPTATIVE TIME CONSTANTS WITH FEEDBACK #######################################

class AFRNN(NN):
    """
        Reccurent neural network model that derives from adaptative continuous time RNN with additional feedback

    """
    def __init__(self, input_size, hidden_size, output_size, alpha):
        super().__init__(input_size, hidden_size, output_size)
        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

        self.feedback_delay = int(50*FS/1000) # from ms to samples
        self.f2h = nn.Linear(output_size, hidden_size, dtype=torch.float32)

        unique_alphas = np.unique(alpha)
        alphas = torch.randint(0, len(unique_alphas),(1,self.hidden_size), dtype=torch.float64) if len(unique_alphas)!=0 else torch.zeros((1,self.hidden_size), dtype=torch.float64)
        for i_alpha, unique_alpha in enumerate(unique_alphas):
            alphas = torch.where(alphas==i_alpha, unique_alpha, alphas)
        alphas = alphas.to(torch.float32)
        self.alpha = nn.Parameter(alphas, requires_grad=True)

    def init_feedback(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.output_size, dtype=torch.float32)

    def get_weights(self):
        return {"B": self.input2h.weight, "J": self.h2h.weight, "W": self.fc.weight, 'A': self.f2h.weight}

    def recurrence(self, input, hidden, feedback):
        h_new = torch.tanh(self.input2h(input) + self.h2h(hidden) + self.f2h(feedback))
        h_new = self.alpha * (h_new - hidden) + hidden
        return h_new

    def forward(self, input, hidden=None, feedback=None):
        """Propogate input through the network."""
        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)
        if feedback is None:
            feedback = self.init_feedback(input.shape).to(input.device)

        # Loop through time
        rnn_output = []
        out = []
        steps = range(input.size(0))
        for i in steps:
            if i < self.feedback_delay:
                hidden = self.recurrence(input[i], hidden, feedback).to(input.device)
            else:
                hidden = self.recurrence(input[i], hidden, out[i-self.feedback_delay]).to(input.device)
            rnn_output.append(hidden)
            out.append(self.fc(hidden))

        # Stack together output from all time steps
        out = torch.stack(out, dim=0)  # (seq_len, batch, hidden_size)
        rnn_output = torch.stack(rnn_output, dim=0)
        return out, rnn_output


class AFEIRNN(AEIRNN):
    """
        Reccurent neural network model that derives from adaptative continuous time Excitatory-Inhibitory RNN with additional feedback
    """

    def __init__(self, input_size, hidden_size, output_size, alpha,
                 e_prop=0.8, sigma_rec=0, **kwargs):
        super().__init__(input_size, hidden_size, output_size, alpha, e_prop=e_prop, sigma_rec=sigma_rec)
        self.feedback_delay = int(50*FS/1000) # from ms to samples
        self.f2h = nn.Linear(output_size, hidden_size, dtype=torch.float32)

    def init_feedback(self, input):
        batch_size = input.shape[1]
        return torch.zeros(batch_size, self.output_size).to(input.device)

    def get_weights(self):
        return {"B": self.input2h.weight, "J": self.h2h.effective_weight(), "W": self.fc.weight, 'A': self.f2h.weight}

    def recurrence(self, input, hidden, feedback):
        """Recurrence helper."""
        state, output = hidden
        total_input = self.input2h(input) + self.h2h(output) + self.f2h(feedback)
        state = self.alpha * (total_input - state) + state
        state += self._sigma_rec * torch.randn_like(state, device=input.device)
        output = torch.tanh(state)
        return state, output

    def forward(self, input, hidden=None, feedback=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input)
        if feedback is None:
            feedback = self.init_feedback(input)

        rnn_activity = []
        out=[]
        steps = range(input.size(0))
        for i in steps:
            if i < self.feedback_delay:
                hidden = self.recurrence(input[i], hidden, feedback)
            else:
                hidden = self.recurrence(input[i], hidden, out[i-self.feedback_delay])
            rnn_activity.append(hidden[1])
            out.append(self.fc(hidden[1][:,:self.e_size]))

        out = torch.stack(out, dim=0)  # (seq_len, batch, hidden_size)
        rnn_output = torch.stack(rnn_output, dim=0)
        return out, rnn_activity

############################## ADAPTATIVE TIME CONSTANTS WITH FEEDBACK AND SYNAPSE ############################


class ASFRNN(NN):
    """
        Reccurent neural network model that derives from adaptative continuous time RNN with additional equation for synaptic transmission
    """

    def __init__(self, input_size, hidden_size, output_size, alpha):
        super().__init__(input_size, hidden_size, output_size)
        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

        self.feedback_delay = int(50*FS/1000) # from ms to samples
        self.f2h = nn.Linear(output_size, hidden_size, dtype=torch.float32)

        unique_alphas = np.unique(alpha)
        alphas = torch.randint(0, len(unique_alphas),(1,self.hidden_size), dtype=torch.float64) if len(unique_alphas)!=0 else torch.zeros((1,self.hidden_size), dtype=torch.float64)
        for i_alpha, unique_alpha in enumerate(unique_alphas):
            alphas = torch.where(alphas==i_alpha, unique_alpha, alphas)
        alphas = alphas.to(torch.float32)
        alpha_s = alphas.detach().clone()

        self.alpha = nn.Parameter(alphas, requires_grad=True)
        self.alpha_s = nn.Parameter(alpha_s, requires_grad=True)

    def init_feedback(self, input):
        batch_size = input.shape[1]
        return torch.zeros(batch_size, self.output_size).to(input.device)

    def get_weights(self):
        return {"B": self.input2h.weight, "J": self.h2h.weight, "W": self.fc.weight, 'A': self.f2h.weight}

    def recurrence(self, input, hidden, synapse, feedback):
        s_new = self.alpha_s*(self.input2h(input) + self.h2h(hidden) + self.f2h(feedback) - synapse) + synapse # feedback comes back in the synapse
        h_new = self.alpha*(torch.tanh(s_new) - hidden) + hidden
        return h_new, s_new

    def forward(self, input, hidden=None, synapse=None, feedback=None):
        """Propogate input through the network."""

        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)
        if synapse is None:
            synapse = self.init_hidden(input.shape).to(input.device)
        if feedback is None:
            feedback = self.init_feedback(input)

        # Loop through time
        rnn_output = []
        out = []
        steps = range(input.size(0))
        for i in steps:
            if i < self.feedback_delay:
                hidden, synapse = self.recurrence(input[i], hidden, synapse, feedback)
            else:
                hidden, synapse = self.recurrence(input[i], hidden, synapse, out[i-self.feedback_delay])
            hidden.to(input.device)
            synapse.to(input.device)
            rnn_output.append(hidden)
            out.append(self.fc(hidden))

        # Stack together output from all time steps
        out = torch.stack(out, dim=0)
        rnn_output = torch.stack(rnn_output, dim=0)
        return out, rnn_output


class ASFEIRNN(AEIRNN):
    """
        Reccurent neural network model that derives from adaptative continuous time Excitatory-Inhibitory RNN with additional feedback
    """

    def __init__(self, input_size, hidden_size, output_size, alpha,
                 e_prop=0.8, sigma_rec=0, **kwargs):
        super().__init__(input_size, hidden_size, output_size, alpha, e_prop=e_prop, sigma_rec=sigma_rec)
        self.feedback_delay = int(50*FS/1000) # from ms to samples
        self.f2h = nn.Linear(output_size, hidden_size, dtype=torch.float32)

        alpha_s = self.alpha.detach().clone()
        self.alpha_s = nn.Parameter(alpha_s, requires_grad=True)

    def init_feedback(self, input):
        batch_size = input.shape[1]
        return torch.zeros(batch_size, self.output_size).to(input.device)

    def get_weights(self):
        return {"B": self.input2h.weight, "J": self.h2h.effective_weight(), "W": self.fc.weight, 'A': self.f2h.weight}

    def recurrence(self, input, hidden, synapse, feedback):
        """Recurrence helper."""
        state, output = hidden
        state = state.to(torch.float32)
        output = output.to(torch.float32)
        s_new = self.alpha_s*(self.input2h(input) + self.h2h(output) + self.f2h(feedback) - synapse) + synapse
        state = self.alpha*(torch.tanh(s_new) - state) + state
        state += self._sigma_rec * torch.randn_like(state, device=input.device)
        output = torch.tanh(state)
        return (state, output), s_new


    def forward(self, input, hidden=None, synapse=None, feedback=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input)
        if synapse is None:
            synapse = self.init_hidden(input)[0].to(input.device)
        if feedback is None:
            feedback = self.init_feedback(input)

        # Loop through time
        rnn_output = []
        out = []
        steps = range(input.size(0))
        for i in steps:
            if i < self.feedback_delay:
                hidden, synapse = self.recurrence(input[i], hidden, synapse, feedback)
            else:
                hidden, synapse = self.recurrence(input[i], hidden, synapse, out[i-self.feedback_delay])
            synapse.to(input.device)
            rnn_output.append(hidden[1])
            rnn_e = hidden[1][:, :self.e_size]
            out.append(self.fc(rnn_e.to(torch.float32)))

        # Stack together output from all time steps
        out = torch.stack(out, dim=0)
        rnn_output = torch.stack(rnn_output, dim=0)
        return out, rnn_output
