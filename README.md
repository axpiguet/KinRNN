# little_RNN

## Environment creation ##

<ol>

<li>download the data and put it in tests_results folder</li>
https://we.tl/t-D7eP3OzxP9

<li>create conda environment with name little_RNN</li>
conda create -n little_RNN anaconda python=3.9

<li>activate environment 
conda activate little_RNN

<li>install sklearn genetics</li>
conda install -c conda-forge sklearn-genetic

<li>install pytorch</li> 
follow instruction on pytorch webpage get started : https://pytorch.org/get-started/locally/ 

<li>update the environment 
conda update --all 

</ol>

## Data ## 

<li> Next to the src folder create a folder named "tests_results". Put your data folder in "tests_results".</li>

Your data should be a dataframe which raws are trials and colums parameters. For each muscle, an associated column should contain the arrays of EMG recordings. 
The data should be pre-filtered and normalized per muscle. 

## Training an RNN ## 

To train an RNN : 

<ol>

<li> Check the constant parameters in the __init__.py file of each package</li>

<li> In the folder src/tests/params_files create a new_parameters file that has the name "constants_{name_of_your_test}" following the demo parameters file.</li>

<li> To lauch the training type in the source folder "python main_rnn_training.py "{name_of_your_test}".</li>

The code will store the results of your test in the folder "tests_results".

</ol>

## Scripts descriptions ##
<li>  Visualize files generate gif displaying stim , emg and kin in sync.</li>
<li>  SeeDatajoints.py generates pictures showing all signals available for every trial.</li>
<li>  Quantify.py generates heatmaps for various data quantifications.</li>
<li>  makeDataset.py generates files containing data for one or several selected trial.</li>
