import tensorflow as tf
import numpy as np

def get_unsupervised_dataset(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 256.0, x_test / 256.0
    y_train = tf.squeeze(tf.one_hot(y_train, depth=10), axis=1)
    y_test = tf.squeeze(tf.one_hot(y_test, depth=10), axis=1)
    # combine train and test X

    x = np.concatenate([x_train, x_test], axis=0)
    x = x.astype('float32')
    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.batch(batch_size)

    return dataset

def get_supervised_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 256.0, x_test / 256.0
    y_train = tf.squeeze(tf.one_hot(y_train, depth=10), axis=1)
    y_test = tf.squeeze(tf.one_hot(y_test, depth=10), axis=1)
    return (x_train, y_train), (x_test, y_test)
