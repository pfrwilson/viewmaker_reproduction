# Viewmaker Networks: A Reproducibility Study

This repository contains a reproduction of the paper [Viewmaker Networks: Learning Views for Unsupervised Representation Learning ](https://arxiv.org/abs/2010.07432). 

## Requirements

We performed our experiments on a ubuntu linux machine with tensorflow version 2.6.0, CUDA version 9.1, and CUDnn version 8.2.4

To install the necessary dependencies, run 
```
pip install -r requirements.txt
```

The command
```
source init_env.sh
```
is necessary to allow absolute imports and run the script files. 

Complete environment setup:
```setup
git clone https://github.com/pfrwilson/viewmaker_reproduction
cd viewmaker_reproduction
conda create -n my_env
conda activate my_env
pip install -r requrements.txt
source init_env.sh
```


## Training

Training is performed in two steps - unsupervised pre-training and supervised transfer learning. The unsupervised pre-training trains an encoder network (together with a trainable viewmaker network) using the sim-clr objective. The transfer learning task is to take the trained encoder and viewmaker, add a classification network on top of the frozen models, and train the classification module to perform the supervised learning task. 

So far, we have provided 4 experiment scripts:
- `sim_clr_pretrain`
- `sim_clr_transfer` - coming soon
- `viewmaker_pretrain` - coming soon
- `viewmaker_transfer` - coming soon

These scripts cover pretraining and transfer learning for the original sim_clr model and for the sim_clr model with additional adversarially trainable viewmaker as presented in the original paper. 

The scripts come equipped with default settings to train on the cifar_10 dataset with the hyperparameters given in the experiments described in appendix 1.A. For pretraining, it is necessary to specify the directory into which the model's weights will be saved, and optionally one can specify a file (.h5 extension) from which to load the model weights (if you have partially trained the model and want to continue). For transfer learning, it is necessary to specify a directory to save the trained classifier and a directory from which to load the pre-trained model weights. To use a different experimental configuration, one can create a json-encoded dictionary file encoding the configuration parameters and pass its location as a command-line argument as well. See the script files for the defaults configuration and make sure to define every parameter in the configuration in the json file.

Example training with defaults:
```
python scripts/sim_clr_pretrain.py --save_filepath 'model_weights.h5' 
```
Example training with custom params:
``` 
python scripts/sim_clr_pretrain.py --save_filepath 'model_weights.h5' --params_filepath 'sample_params.txt'
```

## Evaluation

TBA

.