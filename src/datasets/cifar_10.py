import tensorflow as tf
from src.datasets.data_loader import DataLoader
from src.utils.SimCLR_data_util import preprocess_for_train


CIFAR_10_STATISTICS = {
    'mean': [0.49139968, 0.48215841, 0.44653091],
    'std': [0.24703223, 0.24348513, 0.26158784]
}


class CIFAR10(DataLoader):

    def __init__(self):
        self.data = tf.keras.datasets.cifar10.load_data()

    def get_input_shape(self):
        return 32, 32, 3

    def get_num_classes(self):
        return 10

    def get_dataset_for_pretraining_length(self):
        return len(self.get_dataset_for_pretraining())

    def get_dataset_for_pretraining(self):
        (x_train, _), (_, _) = self.data
        x_train = x_train/255.
        return tf.data.Dataset.from_tensor_slices(x_train)

    def get_dataset(self):
        (x_train, y_train), (x_test, y_test) = self.data
    
        y_train = tf.squeeze(tf.one_hot(y_train, depth=10), axis=1)
        y_test = tf.squeeze(tf.one_hot(y_test, depth=10), axis=1)

        x_train = x_train/255.
        x_test = x_test/255.

        train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        return train, test

    def get_augmentation_layer(self) -> tf.keras.layers.Layer:
        
        class AugmentationLayer(tf.keras.layers.Layer):
            def __init__(self):
                super().__init__()
                self.augmentation = lambda im: preprocess_for_train(
                im,
                32, 
                32, 
                blur=False 
            )

            def call(self, x):
                x = tf.map_fn(self.augmentation, x)
                return x

        return AugmentationLayer()

    def get_preprocessing_layer(self) -> tf.keras.layers.Layer:
        
        class PreprocessingLayer(tf.keras.layers.Layer):
            def __init__(self):
                super().__init__()

            def call(self, x):
                x = (x - CIFAR_10_STATISTICS['mean'])/CIFAR_10_STATISTICS['std']
                return x

        return PreprocessingLayer()
                

        