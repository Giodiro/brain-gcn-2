# Analyzing brain connectivity during sleep 
#### Mélanie Bernhardt
#### Semester project - Research in Data Science Course
#### Supervisor: J. Buhmann, Advisor: D. Miladinovic
#### ETH Zurich - Machine Learning Institute

In this semester project, we show how graph neural models can be applied on multi-frequency brain connectivity data derived from MEG recordings in the context of binary sleep stage classification. 

This repository contains the code associated to this project. It contains all the necessary files to:
 1. Build the numpy feature matrix from the original matlab files
 2. Construct and train the graph classification neural network presented in the report.
 3. Run the experiments described in the report.

This readme details the content of the files of this repo and the functions they contain.
Every function and its arguments is also documented precisely in the code.

## Build feature matrices from original matlab files 
Use the file `build_features.py` to compute and save the features matrices needed for the experiments. 
It contains the following functions:
* `prepare_X` loads and merges the original MatLab files in one single numpy array of shape [nobs, 4095, 50]. It takes a list of subject as input argument to only load data for a subset of subjects if needed. 
* `transform_X_std` performs the standard frequency band aggregation preprocessing step. It takes one matrix of size [nobs, 4095, 50] as input.
* `transform_X_one`performs the 'one' frequency band aggregation preprocessing step.

Running the file will trigger the creation and saving of:
 * the full original feature matrix of shape [nobs, 4095, 50] per subject. Each subject matrix will be saved in the `/matrices/all/SXX.npy` folder, where XX is the number of the subject.
 * the standard frequency band matrix per subject. Each subject matrix will be saved in the `/matrices/std/SXX.npy` folder, where XX is the number of the subject.
 * the one frequency band matrix per subject. Each subject matrix will be saved in the `/matrices/one/SXX.npy` folder, where XX is the number of the subject.
 * the label array per subject. Each label array will be saved in the `/matrices/y/SXX.npy` folder, where XX is the number of the subject.
 
**Please create a matrices folder and a runs folder in this repository folder prior to running anything. 
Please run this file prior to running any other file of this project as the other files assume that the data is saved in a npy array following the folder structure described above.**

## Quick start guide to run the experiments of the report
The `classification_exp.py` is the main file to run the experiments described in the report. In the initialization phase, it creates a subfolder in the `runs` directory where the logger and the eventual plots produced during the run of the file are saved. The estimators are then initialized. Finally all three cross-validation settings (i.e. within-one-single subject, within-all subjects CV and across-subjects CV) are run. All files used to build the experiments are explained in more details in the next sections of this Readme.

#### Console options
Several console arguments are available to change the parameters of the experiments without having to modify the code. 
Run `classification_exp.py` with the following console arguments (optional):
 - `-est` a list of estimators names. Choose among 'uniform' (dummy classifier, random prediction between class 1 and 0), 'constant' (dummy classifier always predicts class 0), 'gcn' (GCN_estimator, the model proposed in the report), 'pcasvm' (PCA+SVM pipeline described as the first baseline in the report), 'rf' (SelectPercentile+RandomForestClassifier pipeline, second baseline in the report). If not specified the file uses ['gcn', 'rf', 'pcasvm']
 - `-up` whether to use upsampling or not. If not specified the file uses 'False'.
 - `-t` type of frequency band aggregation to use. It is mandatory to use 'std' if 'gcn' is in your estimators' list. Otherwise one can also use 'one' aggregation (see note in cross-validation section below).
 - `-j` number of jobs to use for the sklearn baselines.
 Specific options for `gcn` estimator:
 - `-s` number of training steps to use
 - `-h1` dimension of the first hidden layer in the node GCN
 - `-h2` dimension of the second hidden layer in the node GCN
 - `-out` dimension of the node embeddings (i.e. dimension of the output layer of the node GCN)
 
 Some examples of commands:
 * To run all the experiments of the report:
  - First `python classification_exp.py` without any parameters runs the experiments for GCN (with the parameters chosen in the report), RF without upsampling and PCA SVM without upsampling.
  - Second `python classification_exp.py -est rf pcasvm -up True` to run the experiments for the baselines with upsampling. 
 * Another example that would run the experiments just with the Graph neural network using a customized architecture and 300 training steps:
 `python classification_exp.py -est gcn -h1 32 -h2 64 -out 128 -s 300`
 #### Output folder
 Every time the `classification_exp.py` file is run it creates a timestamped (e.g. `24Aug18_165932`) subfolder stored in the `runs` folder. All the results of the cross-validations are saved in this subfolder. This subfolder contains:
 * a `std_long.log` file (where std is the type of matrix used) where all the messages are saved: all the results per fold, all the monitoring messages during the Graph Classification Network training (i.e. validation and training loss every 5 training steps). 
* a `std_short.log` file which saves only the results (i.e. the metrics) in each cross-validation run. More convenient to analyze the results without having to go through all training information.
* Several numpy array named in the pattern `within_SXX_est` where XX is the subject number and est the estimator name. These array correspond to the list of balanced accuracy per fold for each within-one-single subject cross-validation.
* One numpy array `within_mixed_est` with est the estimator name. Contains the list of balanced accuracy per fold for the within-all subject cross-validation.
* One numpy array `across_mixed_est` with est the estimator name. Contains the list of balanced accuracy per fold for the across-subject cross-validation.

These numpy array are used to create the final plots in the report (see iPython notebook).

If `gcn` is included in the estimator list it also contains the train/validation loss and balanced accuracy plots for each fold for each cross-validation setting. 
Example of file names: 
* `_across_testsubj_S04_bal_acc.png`corresponds the balanced accuracy evolution for across-subject cross-validation where the test fold was subject 4. 
* `_within_8_subjects_fold_3_loss.png` the training and validation loss plot for fold 3 in within-all-subjects cross-validation.
* `_within_subj_S05_fold_2_bal_acc.png` balanced accuracy evolution for training and validation set for fold 2 of the within-subject-S05 cross-validation.
 
 
## Graph classification network files
The `siamese_gcn` folder contains all the files necessary to build and train the grah classification network described in the report. It contains 3 files:
* `model.py`contains the convolutional layer definiton, the definition of the node classification network and the definition of the graph classification network. 
* `train_utils.py` defines all core functions to train the network
* `GCN_estimator.py` wraps the constructed network into an object of class BaseEstimator from sklearn, this allows us to use the same procedure with the baseline and with our network. 

### model.py
One layer class:
* Class `GraphConvLayer` defines the graph convolutional layer. The implementation of this class is taken from https://github.com/tkipf/pygcn.

Two network definition class. Each of these classes contains one `init` method and one `forward` method.
* Class `NodeGCN` defines the node graph convolution network to derive the nodes features.
* Class `GraphClassificationNet` defines the whole graph classification network. Combines the node classification, sum pooling, fully connected layer.

### train_utils.py
This file defines the training loop, a single training step and a single validation step. This is the main helper file for
the fit/predict functions in the `GCN_estimator.py` file.

* `training_step`: defines a single training step for one batch of observations.1. Computes the Ahat matrix for each frequency band, for each observation in the batch. 2. Gets the outputs from the network with this input 3. Apply the optimizer 4. Compute the training balanced accuracy over the batch
 * `val_step`: defines a single validation step for the validation set. 1. Computes the Ahat matrix for each frequency band for each observation in the batch. 2. Gets the outputs from the network with this input 3. Compute the validation balanced accuracy and loss over the batch.

* `training_loop` function that runs the training loop and saves the plots of the training and validation loss and balanced accuracy. Assumes that standard frequency band aggregation preprocessing step has been performed for the feature matrix i.e. expects an input matrix of size `[ntrain, 5*nchannels]`. 

### GCN_estimator.py
This file defines a wrapper class for the GCN. This is necessary in order to use this network just as
it was any sklearn estimator in the (custom) cross validation.
It defines the `GCN_estimator_wrapper` class a child of the `BaseEstimator` and `ClassifierMixin` classes. It contains one `init` method, one `fit` method (i.e. training method, calls the training loop function defined in `train_utils.py`), one `predict` method (to predict labels on a test set after having fitted i.e. trained the estimator), one `predict_proba` method (to predict probabilities associated to each class on a test set after having fitted i.e. trained the estimator).

## Building the custom cross-validation procedures from the report
### CV_utils.py
The file `CV_utils.py`  defines all the customized cross validations procedure decribed in the experiments section of the report.

* Class `UpsampleStratifiedKFold` customizes the standard `StratifiedKFold` class available in sklearn in order to add the possiblity to perfrom upsampling on each training fold. It implements the procedure detailed in the report. It contains a `init` method a `get_n_splits` method (gets the number of splits) and a `split` method the split method yields a index iterator with train and test indices for each fold in the CV (cf. `StratifiedKFold` class of sklearn package).

Both custom cross-validation functions take the same input arguments.
* Function `AcrossSubjectCV` runs the across-subject cross-validation experiment of the report. It splits the dataset such that one subject is one fold. Then it runs the cross-validation saving the results to a logger file. 
    - Args:
        * `estimator`: estimator object (child of BaseEstimator) to use
        * `logger`: logger object to print the results to.
        * `subject_list`: list of str with the subject names to include in the CV. For standard across-subject CV use the list of all the subjects.
        * `mat`: type of aggregation for the feature matrix preprocessing. Can choose between 'std' (i.e. standard frequency band aggregation) or 'one' (i.e. single frequency band aggregation). ATTENTION: if your estimator is GCN_estimator you can only use 'std' as feature matrix. You can not use the original matrices without processing (saved in 'all' folder) as they are saved as a 3-dimensional array [nobs, 4095,50], if you want to run classification experiments on the original matrices you have to reshape them.  
        * `upsample`: whether to use upsampling or not.
     - Returns:
       * `results`: dataframe containing the results averaged over all folds
       * `metrics`: dataframe containing the results per fold
       * `confusion`: list of confusion matrix per fold
       * `conf_perc`: list of percentage confusion matrix per fold
    
* Function `WithinOneSubjectCV` implements custom within subject CV. Returns the same type of arguments as the previous function. It takes the same arguments as the previous function. In particular, it takes a list of subject as arguments: if one subject is specified it performs within-one-single subject cross validation (first setting in the report). If a list of all the subjects is passed to the function it performs within-all-subject cross-validation. 

## Plots of the report
The notebook `plot_results_report.ipynb` is the notebook used to create the results plots and latex table of the report. Just modify the run folder names in the second cell of the notebook if you wish to create new results plots with other experiment parameters. 

