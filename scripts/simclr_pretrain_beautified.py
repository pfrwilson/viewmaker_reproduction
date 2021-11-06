
import tensorflow as tf
import tensorflow_addons as tfa
import hydra
from omegaconf import DictConfig
import argparse

from src.models.resnet_small_version2 import ResNet18
from src.models.layers import Identity
from src.datasets import cifar_10
from src.models.SimCLR import SimCLR

CONFIG_PATH:str = '../configs'
CONFIG_NAME:str = 'sample_config'

def build_model(input_shape, temperature, embedding_dim, load_filepath=None):

    encoder = ResNet18(
        input_shape = input_shape,
        classes=10
    )

    encoder.fc = Identity()

    projection_head = tf.keras.layers.Dense(embedding_dim)

    preprocessing_layer = cifar_10.get_preprocessing_layer()

    model = SimCLR(
        encoder, 
        preprocessing_layer, 
        projection_head, 
        temperature=temperature
    )

    model.build(input_shape=input_shape)

    if load_filepath:
        model.load_weights(load_filepath)

    return model

def get_optimizer(optimizer_name):
    if optimizer_name == 'adam':
        return tf.keras.optimizers.Adam
    elif optimizer_name == 'sgd':
        return tfa.optimizers.SGDW
    else: 
        raise Exception(f'{optimizer_name} not recognized')

@hydra.main(config_path = CONFIG_PATH, config_name = CONFIG_NAME)
def train(args: DictConfig) -> None:
    
    dataset = cifar_10.get_unsupervised_dataset(args.batch_size)

    model = build_model(
        (None, 32, 32, 3), 
        args.temperature,
        args.embedding_dim,
        load_filepath = args.load_filepath
    )

    optimizer = get_optimizer(args.optimizer_name)(
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay, 
        momentum = args.momentum
    )
    
    model.compile(optimizer=optimizer)

    model.fit(dataset, epochs=args.epochs)

    model.save_weights(args.save_filepath)


if __name__ == '__main__':
    train()


