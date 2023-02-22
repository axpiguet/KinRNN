from os.path import dirname, abspath, join
import pandas as pd

ROOT_DIR = dirname(abspath(__file__))
PATH = abspath(join(ROOT_DIR, '../../tests_results/'))

# Training
LR = [0.1,0.1]#0.0002,0.0001] #
BATCH_SIZE = [2,32]
GPU = "cuda:0"
ALPHA = pd.DataFrame({13: [0.308], 20: [0.2], 40: [0.1], 60: [0.067], 80: [0.05], 100: [0.04] ,120: [0.034]})#4/freq
WORTH_MP = 200
TEST_SIZE = 0.2


from .save import load_data, load_checkpoint, dump_checkpoint, dump_loss
from .predict import predict
from .train import train, train_per_configs , train_emg
from .interpret import plot, plot_pred, plot_alpha_activity, plot_EI_activity, plot_alpha_distribution, plot_eigenvalues, plot_heatmap, plot_PCA, plot_loss_trajectory
from .models import RNN, LSTM, LSTM2Biomech, LSTM2NoBiomech, LSTMsep, LSTM2, GRU, ARNN, AEIRNN, ASRNN, ASEIRNN,ffEIRNN,EIRNN, AFRNN, AFEIRNN, ASFRNN, ASFEIRNN , FffEIRNN
