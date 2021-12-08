from abc import ABC, abstractmethod
import tensorflow as tf

from src.datasets.cifar_10 import CIFAR10
from src.datasets.mnist import MNIST
from src.datasets.speech_commands import SpeechCommands

DATASETS = {
    'cifar_10': CIFAR10,
    'mnist': MNIST,
    'speech_commands': SpeechCommands
}


def get_data_loader(dataset_name):
    """
    Factory method for getting a DataLoader instance
    """

    if dataset_name in DATASETS.keys():
        return DATASETS[dataset_name]()

    else:
        raise NotImplementedError(f'Dataset {dataset_name} not supported.')