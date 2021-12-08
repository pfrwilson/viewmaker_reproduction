import tensorflow as tf
from src.datasets.data_loader import DataLoader
from einops import rearrange

MNIST_STATISTICS = {
    'mean': 0.1306604762738429,
    'std': 0.3081078038564622
}


class MNIST(DataLoader):

    def __init__(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        y_train = tf.one_hot(y_train, depth=10)
        y_test = tf.one_hot(y_test, depth=10)
        x_train = x_train / 255.
        x_train = rearrange(x_train, ' b h w -> b h w 1', )
        x_test = rearrange(x_test, ' b h w -> b h w 1',)
        x_test = x_test / 255.
        self.data = (x_train, y_train), (x_test, y_test)

    def get_input_shape(self):
        return 28, 28, 1

    def get_num_classes(self):
        return 10

    def get_dataset_for_pretraining(self):
        (x_train, _), (_, _) = self.data
        return tf.data.Dataset.from_tensor_slices(x_train)

    def get_dataset(self):  
        (x_train, y_train), (x_test, y_test) = self.data

        train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        return train, test

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

