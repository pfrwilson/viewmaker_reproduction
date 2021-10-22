import tensorflow as tf
import tensorflow_addons as tfa

from argparse import ArgumentParser
import os

from src.datasets.cifar_10 import get_unsupervised_dataset
from src.models.resnet_small import ResNet18
from src.models.SimCLR import SimCLR
from src.utils.SimCLR_data_util import preprocess_for_train

# default hyper parameters as used by authors
DEFAULT_PARAMS = {
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
    parser.add_argument('--load_filepath', type=str, default=None)
    parser.add_argument('--params_filepath', type=str, default=None, help='json file specifying hyperparameters')
    args = parser.parse_args()
    return args


def get_encoder(params):
    encoder_base = ResNet18(10)
    encoder = tf.keras.Sequential(
        encoder_base.layers[:-1],
        name='encoder'
    )
    return encoder


def get_projection_head(params):
    # no additional projection head
    return tf.keras.Sequential(
        [tf.keras.layers.Dense(params['embedding_dim'], activation=None)],
        name='projection_head'
    )


def get_viewmaker(params):
    class MyViewmaker(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__(name='expert_augmentation_viewmaker')
        def call(self, x):
            augment_image = lambda im: preprocess_for_train(im, 32, 32)
            return tf.map_fn(augment_image, x)
    return MyViewmaker()

def get_dataset(params):
    dataset = get_unsupervised_dataset(batch_size=params['batch_size'])
    return dataset

def run_training(params, args):

    dataset = get_dataset(params)
    encoder = get_encoder(params)
    viewmaker = get_viewmaker(params)
    projection_head = get_projection_head(params)
    
    model = SimCLR(encoder, 
        viewmaker, 
        projection_head, 
        temperature=params['temperature'],
        name='SimCLR_model'
    )
    model(next(iter(dataset))) #build model by calling on batch - necessary for loading weights
    print(model.summary())

    if args.load_filepath: 
        model.load_weights(args.load_filepath)

    optimizer = tfa.optimizers.SGDW(
        learning_rate = params['learning_rate'],
        momentum = params['momentum'],
        weight_decay = params['weight_decay']
    )
    model.compile(optimizer=optimizer)

    model.fit(dataset, epochs=params['epochs'])

    filepath = os.path.join(args.save_dir, 'model_weights.h5')
    model.save_weights(filepath)

if __name__ == '__main__':

    args = get_args()
    if args.params_filepath:
        import json
        with open(args.params_filepath) as params_json:
            params = json.load(params_json)
    else:
        params = DEFAULT_PARAMS

    run_training(params, args)

