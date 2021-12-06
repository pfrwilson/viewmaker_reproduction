from abc import ABC, abstractmethod
import tensorflow as tf

from src.datasets.cifar_10 import CIFAR10
from src.datasets.mnist import MNIST


def get_data_loader(dataset_name):
    """
    Factory method for getting a DataLoader instance
    """

    if dataset_name == 'cifar_10':
        return CIFAR10()

    elif dataset_name == 'mnist':
        return MNIST()
    
    else:
        raise ValueError(f'datasets {dataset_name} cannot be found.')


