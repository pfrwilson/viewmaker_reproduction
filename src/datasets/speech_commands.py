from src.datasets.data_loader import DataLoader
import tensorflow as tf
import tensorflow_datasets as tfds


class SpeechCommands(DataLoader):

    def __init__(self):
        self.data = tfds.load('speech_commands')

