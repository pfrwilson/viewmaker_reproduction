import tensorflow as tf
import numpy as np

def get_unsupervised_dataset(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.squeeze(tf.one_hot(y_train, depth=10), axis=1)
    y_test = tf.squeeze(tf.one_hot(y_test, depth=10), axis=1)

    x = x_train
    #x = x.astype('float32')
    mean = x.mean()
    std = x.std()
    x = (x - mean)/std

    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.batch(batch_size)

    return dataset

def get_supervised_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    mean = x_train.mean()
    std = x_train.std()

    x_train = (x_train - mean)/std
    x_test = (x_test - mean)/std
    y_train = tf.squeeze(tf.one_hot(y_train, depth=10), axis=1)
    y_test = tf.squeeze(tf.one_hot(y_test, depth=10), axis=1)

    return (x_train, y_train), (x_test, y_test)
