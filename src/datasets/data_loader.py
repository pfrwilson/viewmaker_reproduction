from abc import ABC, abstractmethod
import tensorflow as tf


class DataLoader(ABC):

    @abstractmethod
    def get_input_shape(self):
        """
        return the shape of a training example X eg (32, 32, 3
        """

    def get_num_classes(self):
        """"
        return the number of classes for the training example y.
        """

    @abstractmethod
    def get_dataset_for_pretraining(self):
        """
        Return a tf.data.Dataset child class which generates data examples X,
        for unsupervised pretraining.
        """

    @abstractmethod
    def get_dataset(self):
        """
        Return a pair of datasets train, test, where both train and test subclass tf.data.Dataset
        train should return pairs X, y where batches of X can be passed to either the
        augmentation layer or preprocessing layer. Note that y must be encoded as one-hot classes.
        """

    @abstractmethod
    def get_augmentation_layer(self) -> tf.keras.layers.Layer:
        """
        Augmentation layer appropriate for dataset.
        Inputs and outputs should be in the range (0,1)
        """

    @abstractmethod
    def get_preprocessing_layer(self) -> tf.keras.layers.Layer:
        """
        A final preprocessing layer before the data is passed to the
        encoder. This should center and normalize the pixel values to mean 0
        and std 1.
        """

