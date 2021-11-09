
import tensorflow as tf
import tensorflow_addons as tfa
import hydra
from omegaconf import DictConfig

from src.models.resnet_small_version2 import ResNet18
from src.models.layers import Identity
from src.datasets import cifar_10
from src.models.SimCLR import SimCLR_adversarial
from src.models.viewmaker_new import Viewmaker

CONFIG_PATH:str = '../configs'
CONFIG_NAME:str = 'viewmaker_pretrain_config.yaml'


def build_model(input_shape, temperature, embedding_dim, load_filepath=None, 
viewmaker_loss_weight=1.0, distortion_budget=0.05):

    encoder = ResNet18(
        input_shape = input_shape,
        classes=10
    )

    encoder.fc = Identity()

    projection_head = tf.keras.layers.Dense(embedding_dim)

    preprocessing_layer = cifar_10.get_preprocessing_layer()

    viewmaker = Viewmaker(
        num_channels = input_shape[-1],
        distortion_budget = distortion_budget
    )

    model = SimCLR_adversarial(
        encoder, 
        preprocessing_layer, 
        viewmaker, 
        projection_head,
        temperature=temperature, 
        viewmaker_loss_weight = viewmaker_loss_weight
    )

    model.build(input_shape=input_shape)

    if load_filepath:
        model.load_weights(load_filepath)

    return model


@hydra.main(config_path = CONFIG_PATH, config_name = CONFIG_NAME)
def train(args: DictConfig) -> None:
    
    dataset = cifar_10.get_unsupervised_dataset(args.batch_size)

    model = build_model(
        (None, 32, 32, 3), 
        args.temperature,
        args.embedding_dim,
        load_filepath = args.load_filepath
    )

    optimizer = tfa.optimizers.SGDW(
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay, 
        momentum = args.momentum
    )
    
    model.compile(optimizer=optimizer)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.log_directory)

    model.fit(dataset, epochs=args.epochs, callbacks=[tensorboard_callback])

    model.save_weights(args.save_filepath)


if __name__ == '__main__':
    train()


