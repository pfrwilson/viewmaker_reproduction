import tensorflow as tf
import tensorflow_addons as tfa

from argparse import ArgumentParser
import os

from src.datasets.cifar_10 import get_unsupervised_dataset
from src.models.resnet_small_version2 import ResNet18
from src.models.SimCLR import SimCLR
#from src.models.Version2_SimCLR import SimCLR
from src.utils.SimCLR_data_util import preprocess_for_train
from src.models.layers import Identity

# default hyper parameters as used by authors
#DEFAULT_PARAMS = {
#    'temperature': 0.07,
#    'batch_size': 256,
#    'epochs': 200,
#    'learning_rate': 0.03,
#    'momentum': 0.9,
#    'weight_decay': 1e-4, 
#    'embedding_dim': 128
#}
# hyper parameters used by wangxin0716/SimCLR-CIFAR10 
DEFAULT_PARAMS = {
    'temperature': 0.5,
    'batch_size': 512,
    'epochs': 1000,
    'learning_rate': 0.6, 
    'momentum': 0.9, 
    'weight_decay': 1e-6,
    'embedding_dim': 128
}


# command line args
def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_filepath', type=str, help='where to save the model weights')
    parser.add_argument('--load_filepath', type=str, default=None)
    parser.add_argument('--params_filepath', type=str, default=None, help='json file specifying hyperparameters')
    args = parser.parse_args()
    return args

def get_encoder():
    encoder = ResNet18(input_shape=(None, 32, 32, 3), classes=10)
    encoder.fc = Identity()
    return encoder

def get_projection_head(embedding_dim):
    # no additional projection head
    return tf.keras.Sequential([
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(embedding_dim, activation=None)
        ],
        name='projection_head'
    )

def get_viewmaker():
    class MyViewmaker(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__(name='expert_augmentation_viewmaker')
        def call(self, x):
            augment_image = lambda im: preprocess_for_train(im, 32, 32)
            return tf.map_fn(augment_image, x)
    return MyViewmaker()

def get_model(embedding_dim, temperature=1.0, input_shape=(None, 32, 32, 3)):
    encoder = get_encoder()
    viewmaker = get_viewmaker()
    projection_head = get_projection_head(embedding_dim)
    
    model = SimCLR(encoder, 
        viewmaker, 
        projection_head, 
        temperature=temperature,
        name='SimCLR_model'
    )

    model.build(input_shape)

    return model

def get_dataset(batch_size):
    dataset = get_unsupervised_dataset(batch_size=batch_size)
    return dataset

def run_training(params, args):

    dataset = get_dataset(params['batch_size'])
    model = get_model(params['embedding_dim'], temperature=params['temperature'])

    print(model.summary())

    if args.load_filepath: 
        model.load_weights(args.load_filepath)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate = params['learning_rate'], 
        decay_steps = params['epochs']*len(dataset),
        alpha = 1e-3
    )

    optimizer = tfa.optimizers.SGDW(
        learning_rate = lr_schedule,
        momentum = params['momentum'],
        weight_decay = params['weight_decay']
    )
    #optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer)

    model.fit(dataset, epochs=params['epochs'])

    model.save_weights(args.save_filepath)

if __name__ == '__main__':

    args = get_args()
    if args.params_filepath:
        import json
        with open(args.params_filepath) as params_json:
            params = json.load(params_json)
    else:
        params = DEFAULT_PARAMS

    run_training(params, args)

