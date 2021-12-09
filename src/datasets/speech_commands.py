from src.datasets.data_loader import DataLoader
import tensorflow as tf
import tensorflow_datasets as tfds
import scipy
from scipy.signal import spectrogram
import numpy as np


class SpeechCommands(DataLoader):

    def __init__(self):
        self.data = tfds.load('speech_commands')

    def get_input_shape(self):
        return 64, 64, 1

    def get_num_classes(self):
        return 10

    def get_dataset_for_pretraining_length(self):
        return 85511

    def get_dataset_for_pretraining(self):
        def datagen():
            for example in self.data['train']:
                signal = example['audio']
                image = self.convert_to_spectrogram(signal)
                yield image
        output_type = tf.float32
        output_shape = tf.TensorShape((129, 71, 1))
        return tf.data.Dataset.from_generator(datagen, output_types=output_type, output_shapes=output_shape)

    def get_dataset(self):
        def preprocess(example):
            signal = example['audio']
            label = example['label'].numpy()
            image = self.convert_to_spectrogram(signal)
            label = tf.one_hot(label-1, depth=11)
            return signal, label

        def train_datagen():
            for example in self.data['train']:
                yield preprocess(example)

        def test_datagen():
            for example in self.data['test']:
                yield preprocess(example)

        ot = tf.float32, tf.float32
        os = tf.TensorShape((129, 71, 1)), tf.TensorShape((11,))
        train = tf.data.Dataset.from_generator(train_datagen(), output_types=ot, output_shapes=os)
        test = tf.data.Dataset.from_generator(test_datagen(), output_types=ot, output_shapes=os)

        return train, test

    def get_augmentation_layer(self) -> tf.keras.layers.Layer:
        raise NotImplementedError

    def get_preprocessing_layer(self) -> tf.keras.layers.Layer:
        return tf.keras.layers.Resizing(64, 64)

    @staticmethod
    def convert_to_spectrogram(signal):
        image = scipy.signal.spectrogram(signal)[2]
        image = np.expand_dims(image, 2)  # add channel dim
        return image
