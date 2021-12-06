import tensorflow as tf
from src.datasets.data_loader import DataLoader

MNIST_STATISTICS = {
    'mean': 0.1306604762738429,
    'std': 0.3081078038564622
}


class MNIST(DataLoader):

    def get_dataset(self):  
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        y_train = tf.one_hot(y_train, depth=10)
        y_test = tf.one_hot(y_test, depth=10)

        x_train = x_train/255.
        x_test = x_test/255.        

    def get_augmentation_layer(self) -> tf.keras.layers.Layer:
        # TODO make augmentation layer
        raise NotImplementedError()

    def get_preprocessing_layer(self) -> tf.keras.layers.Layer:
        
        class PreprocessingLayer(tf.keras.layers.Layer):
            def __init__(self) -> None:
                super().__init__()

            def call(self, x):
                x = (x - MNIST_STATISTICS['mean'])/MNIST_STATISTICS['std']
                return x

        return PreprocessingLayer()

