import tensorflow as tf
import tensorflow_addons as tfa

from argparse import ArgumentParser
import os

from src.datasets.cifar_10 import get_unsupervised_dataset
from src.models.resnet_small import ResNet18
from src.models.SimCLR import SimCLR
from src.utils.SimCLR_data_util import preprocess_for_train

# default hyper parameters as used by authors
PARAMS = {
    'temperature': 0.07,
    'batch_size': 256,
    'epochs': 200,
    'learning_rate': 0.03,
    'momentum': 0.9,
    'weight_decay': 1e-4, 
    'embedding_dim': 128
}

# command line args
def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='.', help='where to save the model weights')
    parser.add_argument('--epochs', type=int, default=200, help='')
    args = parser.parse_args()
    return args

ARGS = get_args()


def get_encoder():
    encoder_base = ResNet18(10)
    encoder = tf.keras.Sequential(
        encoder_base.layers[:-1]
    )
    return encoder

def get_projection_head():
    # no additional projection head
    return tf.keras.Sequential([
        tf.keras.layers.Dense(PARAMS['embedding_dim'], activation=None)
    ])

def get_viewmaker():
    class MyViewmaker(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
        def call(self, x):
            augment_image = lambda im: preprocess_for_train(im, 32, 32)
            return tf.map_fn(augment_image, x)
    return MyViewmaker()

def run_training():
    dataset = get_unsupervised_dataset(batch_size=PARAMS['batch_size'])
    # model needs batch of EXACTLY batch_size -- leave out last batch:
    dataset = dataset.take(len(dataset)-1) 
    
    encoder = get_encoder()
    viewmaker = get_viewmaker()
    projection_head = get_projection_head()
    
    model = SimCLR(encoder, viewmaker, projection_head, PARAMS['temperature'])

    optimizer = tfa.optimizers.SGDW(
        learning_rate = PARAMS['learning_rate'],
        momentum = PARAMS['momentum'],
        weight_decay = PARAMS['weight_decay']
    )
    model.compile(optimizer=optimizer)

    model.fit(dataset, epochs=ARGS.epochs)

    filepath = os.path.join(ARGS.save_dir, 'model_weights.h5')
    model.save_weights(filepath)

if __name__ == '__main__':
    run_training()

