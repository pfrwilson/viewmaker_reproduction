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
python -m venv venv
venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(cwd)
```

## Training and Evaluation

Training is performed in two steps - unsupervised pre-training and supervised transfer learning.
The unsupervised pre-training trains an encoder network (together with a trainable viewmaker network) 
using the sim-clr objective. The transfer learning task is to take the trained encoder and
viewmaker, add a classification network on top of the frozen models, and train the classification 
module to perform the supervised learning task. 

we provide scripts for pretraining and transfer learning in the `scripts` directory :

`viewmaker_pretrain.py`, `viewmaker_transfer.py`

To run training, we need to provide the experimental configurations in the yaml
file `configs/viewmaker_expt_config.yaml`. The file comes with default model and
hyperparameter settings used in the original paper. To run an experiment, we need only
provide a directory in which to record logs and save weights. To do this, make a directory 
and provide the full path in line 37 ('experiment_directory') of the yaml.

We currently support a limited number of datasets. We can choose the dataset by passing a
dataset name in the config file. The full list of supported datasets can be found in the
`datasets/data_loader_factory.py` module. To add a new dataset, simply subclass the
`datasets.dataloader.DataLoader` base class and add the dataset name and to the
`datasets/dataset_loader_factory.py` file with the appropriate import. Then, change the dataset
name in the config file.  

Once we have provided an experiment directory, we can run the training via 
```bash
python scripts/viewmaker_pretrain.py 
python scripts/viewmaker_trainser.py
```

In the experiment directory, a log directory for the pretraining and transfer learning
will be created that can be opened with tensorboard. A `.h5` file will be created to
save the weights for the encoder, viewmaker, and classifier networks. The configuration
file will be copied into the experiment directory for documentation, and a file `evaluation_log.txt'
will be created measuring the final performance of the classifier. 


