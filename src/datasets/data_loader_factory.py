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
    
    else: raise ValueError(f'datasets {dataset_name} cannot be found.')


class DataLoader(ABC):

    @abstractmethod
    def get_dataset(self):
        """
        Return dataset in the form (x_train, y_train), (x_test, y_test).
        pixel values normalized to the range 0, 1 for image data
        """
        pass

    @abstractmethod
    def get_augmentation_layer(self) -> tf.keras.layers.Layer:
        """
        Augmentation layer appropriate for dataset. 
        Inputs and outputs should be in the range (0,1)
        """
        pass

    @abstractmethod
    def get_preprocessing_layer(self) -> tf.keras.layers.Layer:
        """
        A final preprocessing layer before the data is passed to the
        encoder. This should center and normalize the pixel values to mean 0
        and std 1.
        """
        pass