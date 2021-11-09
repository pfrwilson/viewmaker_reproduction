
import hydra
from omegaconf.dictconfig import DictConfig
import tensorflow as tf
import tensorflow_addons as tfa

from src.models.layers import Identity
from src.models.transfer_learning import TransferModel
from scripts.simclr_pretrain import build_model as build_pretrained_simclr
from src.datasets.cifar_10 import get_supervised_dataset, get_unsupervised_dataset

CONFIG_PATH:str = '../configs'
CONFIG_NAME:str = 'simclr_transfer_config'

def build_model(input_shape, temperature, embedding_dim, load_filepath, load_simclr_filepath, num_classes):
    
    pretrained_model = build_pretrained_simclr(
        input_shape, 
        temperature, 
        embedding_dim, 
        load_filepath=load_simclr_filepath
    )

    encoder = pretrained_model.encoder
    encoder.pool = Identity()               # encoder output is pre-pool 4x4x512 feature map
    preprocessing = pretrained_model.preprocessing_layer
    classifier = tf.keras.Sequential([
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(num_classes),
        tf.keras.layers.Softmax()
    ])
    model = TransferModel(
        encoder, 
        classifier, 
        preprocessing
    )

    model.build(input_shape=input_shape)

    if load_filepath:
        model.load_weights(load_filepath)

    return model

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(args: DictConfig) -> None:
    
    (x_train, y_train), (x_test, y_test) = get_supervised_dataset()

    model = build_model(
        input_shape = (None, 32, 32, 3),
        temperature = args.temperature,
        embedding_dim = args.embedding_dim, 
        load_filepath = args.load.pretrained_classifier_weights, 
        load_simclr_filepath = args.load.pretrained_simclr_weights,
        num_classes = 10
    )

    optimizer = tfa.optimizers.SGDW(
        weight_decay = args.weight_decay,
        learning_rate = args.learning_rate,
        momentum = args.momentum
    )

    model.compile(
        optimizer, 
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.log_dir, histogram_freq=1)

    if args.train:
        
        model.fit(
            x=x_train, 
            y=y_train, 
            batch_size = args.batch_size,
            epochs=args.epochs,
            callbacks=[tensorboard_callback],
            validation_split=0.2
        )

        model.save_weights(args.save.classifier_weights)

    model.evaluate(
        x = x_test, 
        y = y_test, 
        batch_size = args.eval_batch_size
    )


if __name__ == '__main__':
    main()

