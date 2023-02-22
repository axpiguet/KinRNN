#! usr/bin/bash

#PATH='C:/Users/yes/Documents/Github/little_RNN/tests_results/'
#PATH='D:/Documents/NeuroRestore/Code/Github/little_RNN/tests_results/'

PATH=$(pwd)
PATH_PYTHON=/home/$(echo $USER)/anaconda3/envs/little_RNN/bin/python3

arg_list=("rnn" "afrnn" "aeirnn" "asfrnn" "aseirnn" "asfeirnn" "lstm2" "arnn_narrow" "arnn_regularized" "asrnn_narrow" "asrnn_regularized" "afrnn_narrow" "afrnn_regularized" "aeirnn_narrow" "aeirnn_regularized" "gru_narrow" "gru_regularized" "lstm_narrow" "lstm_regularized")

echo $"Training of ${#arg_list[@]} networks..."
echo 
echo

cd $PATH

for i in ${arg_list[@]}
do 
    echo "Training " $i 
    $PATH_PYTHON main_rnn_training.py $i
done

echo "End of script"