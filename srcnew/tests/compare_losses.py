import matplotlib.pyplot as plt 
import _pickle as cPickle 
import numpy as np
import params_files.stable_constants as stable_constants

# passing arguments 
# parser = argparse.ArgumentParser(description='Plot loss of many trainings on the same graph')
# parser.add_argument('IDs', metavar='N', type=list, help='IDs of the tests')
# args = vars(parser.parse_args())
plt.style.use('dark_background')

# ID = args['IDs']
training_configs = 'cath10_an9_freq20'
IDs = ['test_rnn', 'test_asrnn', 'test_asfrnn', 'test_asfeirnn', 'test_lstm', 'test_gru']
names = ['RNN', 'ARU', 'ASFRNN', 'ASFEIRNN', 'LSTM', 'GRU']

color_list = ['#FA525B','#2D5983','#372367','#19D3C5','#B8E600','#FFB11E','#F0DF0D']

def plot_loss_trajectory(training_config, IDs, names)->None:
    """
    plot loss trajectory of the network 

    :param training_config: int defining configuration or String 'main'
    """ 
    fig = plt.figure()  
    #plt.xscale("log")
    for i_config, config in enumerate(IDs):
        losses = cPickle.load(open(stable_constants.PATH + config + f"/{'Sub' if training_config!='main' else 'Main'}-training{(' ' + str(training_config)) if training_config!='main' else ''}/Loss.pkl", "rb" ))
        #cubic_interploation_model = scipy.interpolate.interp1d(np.arange(0, len(losses), 10), losses[::10])
        #Y = cubic_interploation_model(np.arange(0, len(losses)))
        # X_Y_Spline = scipy.interpolate.make_interp_spline(np.arange(0, len(losses), 200), losses[::200])
        # Y = X_Y_Spline(np.arange(len(losses)))
        Y =  losses[::7]
        plt.plot(np.arange(0, len(losses),7), Y, label=names[i_config], color=color_list[i_config])
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.box(False)
    plt.legend()
    #plt.show()
    plt.savefig("loss.png", transparent=True)
    plt.close(fig)

plot_loss_trajectory(training_configs, IDs, names)