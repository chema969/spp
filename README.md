# spp

This program offer a fast and easy way to realise deep learning experiments with TensorFlow, specially focused in ordinal class databases.  

## Program structure

 - **Activation**: Activation layers
 - **Callback**: Class that defines the callback function for the execution
 - **Dataset**: Datasets included in the program
 - **Experiment**: File that run each experiment
 - **Experimentset**: File in which all the experiment are runs
 - **Generators**:File to create generator to perform the fit
 - **Layers**:Auxiliar layers
 - **Losses**: Loss functions
 - **Main_experiment**: Main program for experimentation
 - **Metrics**:Diferent metrics
 - **Net_keras**: Keras network architectures
 - **Resnet**: Resnet tools used in some of the networks.




## Execution
The execution mode is:

> python main_experiment.py experiment --file/-f (experiment_file)  --gpu/-g (gpu_number)

To use this program, a .json file defining the experiment to use is needed. The .json file should contain, at least, one experiment to run. The parameters for each experiment are:
| Name             | Explanation                                                                                                     | Possible values                                                                                                                                                                                                                                              | Default value              |
|------------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| db               | Database to be used                                                                                             | 'cifar10','cifar100',<br>'cinic10','mnist','wiki',<br>'imdb','retinopathy',<br>'adience','historical',<br>'fgnet','fashion_mnist'                                                                                                                            | 'cifar10'                  |
| net_type         | Net architecture to<br>be used                                                                                  | 'vgg19','vgg16','conv128',<br>'inceptionresnetv2',<br>'beckhamresnet'                                                                                                                                                                                        | 'vgg19'                    |
| batch_size       | Batch size                                                                                                      | Any integer value over 0                                                                                                                                                                                                                                     | 128                        |
| epochs           | Number of epochs to <br>train                                                                                   | Any integer value over 0                                                                                                                                                                                                                                     | 100                        |
| checkpoint_dir   | File where checkpoint<br>will be saved                                                                          | Any file name                                                                                                                                                                                                                                                | 'results'                  |
| loss             | Loss function used                                                                                              | 'categorical_crossentropy',<br>'qwk','msqwk'                                                                                                                                                                                                                 | 'categorical_crossentropy' |
| activation       | Activation function<br>used inside the net                                                                      | 'relu', 'lrelu','prelu',<br>'elu','softplus','spp',<br>'sppt','mpelu','rtrelu',<br>'rtprelu','pairedrelu',<br>'erelu','eprelu','sqrt',<br>'rrelu','pelu','slopedrelu',<br>'ptelu','antirectifier','crelu',<br>any other activation suported by<br>tensorflow | 'relu'                     |
| final_activation | Activation for the final<br>layer                                                                               | 'poml','pomp','pomclog',<br>'binomial','pomglogit',<br>'clmcauchit','clmggamma',<br>'clmgauss','clmexpgauss',<br>any other activation suported by<br>tensorflow                                                                                              | 'softmax'                  |
| use_tau          | Allow the use of tau in<br>Proportional Odds Model                                                              | True or False                                                                                                                                                                                                                                                | False                      |
| prob_layer       | Probability layer                                                                                               | 'geometric'                                                                                                                                                                                                                                                  | None                       |
| spp_alpha        | Alpha value in case <br>you use spp                                                                             | Float number between 1 and 0                                                                                                                                                                                                                                 | 0                          |
| lr               | Learning Rate                                                                                                   | Float number between 1 and 0                                                                                                                                                                                                                                 | 0.1                        |
| momentum         | Momentum used                                                                                                   | Float number between 1 and 0.9                                                                                                                                                                                                                                 | 0                          |
| dropout          | Percentage of dropout<br>used in dropout layers                                                                 | Float number between 1 and 0                                                                                                                                                                                                                                 | 0                          |
| task             | Task to be done between<br>train, test or both                                                                  | 'train','test','both'                                                                                                                                                                                                                                        | 'both'                     |
| workers          | Number of threads to be <br>used. If the value is 0,<br>the program will be <br>executed in the main <br>thread | Any integer bigger or equal to 0                                                                                                                                                                                                                             | 4                          |
| queue_size       | Maximum size for the <br>generator queue                                                                        | Any integer value over 0                                                                                                                                                                                                                                     | 1024                       |
| augmentation     | Data augmentation parameters                                                                                    | Data augmentation parameters                                                                                                                                                                                                                                 | {}                         |
| val_type         | Type of validation used                                                                                         | 'holdout' or 'kfold'                                                                                                                                                                                                                                         | 'holdout'                  |
| holdout          | Percentage of validation <br>hold-out                                                                           | Float number between 1 and 0                                                                                                                                                                                                                                 | 0.2                        |
| executions       | In case of holdout,number<br> of executions to do                                                           | Integer number over 1                                                                                                                                                                                                                                  | None             |
| n_folds          | Number of folds to be used                                                                                      | Any integer value over 0                                                                                                                                                                                                                                     | 5                          |
| encode           | Tipe of encode used in the class labels                                                                        |'one_hot','soft_ordinal'                                                                                                                                                                                                                                      | 'one_hot'                |





