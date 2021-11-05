import tensorflow as tf
import numpy as np
from src.utils.SimCLR_data_util import preprocess_for_train

CIFAR_10_STATISTICS = {
    'mean': [0.49139968, 0.48215841, 0.44653091],
    'std': [0.24703223, 0.24348513, 0.26158784]
}


def get_unsupervised_dataset(batch_size):
    """Returns dataset normalized to range 0, 1"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.squeeze(tf.one_hot(y_train, depth=10), axis=1)
    y_test = tf.squeeze(tf.one_hot(y_test, depth=10), axis=1)

    x_train = x_train/255.
    x_test = x_test/255.

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.batch(batch_size)
    dataset = dataset.take(len(dataset)-1)

    return dataset


def get_supervised_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train/255.
    x_test = x_test/255.

    y_train = tf.squeeze(tf.one_hot(y_train, depth=10), axis=1)
    y_test = tf.squeeze(tf.one_hot(y_test, depth=10), axis=1)

    return (x_train, y_train), (x_test, y_test)


def get_preprocessing_layer(use_augmentations=True):
    class MyViewmaker(tf.keras.layers.Layer):

        def __init__(self, use_augmentations=True):
            super().__init__(name='expert_augmentation_viewmaker')
            self.use_augmentations = use_augmentations
            self.augmentations = lambda im: preprocess_for_train(
                im,
                32, 
                32, 
                blur=False 
            )

        def call(self, x, training=False):
            if self.use_augmentations and training:
                x = tf.map_fn(self.augmentations, x)
            x = (x - CIFAR_10_STATISTICS['mean'])/CIFAR_10_STATISTICS['std']
            return x

    return MyViewmaker(use_augmentations=use_augmentations)

