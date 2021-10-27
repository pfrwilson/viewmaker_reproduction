import tensorflow as tf

from argparse import ArgumentParser
import os

from sim_clr_pretrain import get_model
from src.datasets.cifar_10 import get_unsupervised_dataset
from src.models.transfer_learning import get_transfer_classifier

DEFAULT_PARAMS = {
    # model architecture
    'temperature': 0.07,
    'embedding_dim': 128,
    # training
    'batch_size': 256,
    'epochs': 200,
    'learning_rate': 0.03,
    'momentum': 0.9,
    'weight_decay': 1e-4,    
}

# command line args
def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='.', help='where to save the model weights')
    parser.add_argument('--load_filepath', type=str, default=None)
    parser.add_argument('--params_filepath', type=str, default=None, help='json file specifying hyperparameters')
    args = parser.parse_args()
    return args

    #TODO
    #TODO