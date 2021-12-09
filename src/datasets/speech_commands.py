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
                image, label = SpeechCommands._preprocess(example)
                yield image
        output_type = tf.float32
        output_shape = tf.TensorShape((129, 71, 1))
        return tf.data.Dataset.from_generator(datagen, output_types=output_type, output_shapes=output_shape)

    def get_dataset(self):

        def train_datagen():
            for example in self.data['train']:
                yield SpeechCommands._preprocess(example)

        def test_datagen():
            for example in self.data['test']:
                yield SpeechCommands._preprocess(example)

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
    def _convert_to_spectrogram(signal):
        image = scipy.signal.spectrogram(signal)[2]
        image = np.expand_dims(image, 2)  # add channel dim
        return image

    @staticmethod
    def _pad_to(signal, length):
        diff = len(signal) - length
        if diff > 0:
            signal = np.pad(signal, (0, diff))
        return signal

    @staticmethod
    def _preprocess(example):
        signal = example['audio']
        signal = SpeechCommands._pad_to(signal, 60000)
        label = example['label'].numpy()
        image = SpeechCommands._convert_to_spectrogram(signal)
        label = tf.one_hot(label - 1, depth=11)
        return image, label

