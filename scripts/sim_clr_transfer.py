import tensorflow as tf
import tensorflow_addons as tfa

from argparse import ArgumentParser
import os

from scripts.sim_clr_pretrain import get_model
from src.datasets.cifar_10 import get_supervised_dataset
from src.models.transfer_learning import get_transfer_classifier

DEFAULT_PARAMS = {
    # model architecture
    'temperature': 0.07,
    'embedding_dim': 128,
    # training
    'batch_size': 128,
    'epochs': 100,
    'learning_rate': 0.01,
    'momentum': 0.9,
    'weight_decay': 0,    
}

# command line args
def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_filepath', type=str, default='model_weights.h5', help='where to save the model weights')
    parser.add_argument('--load_filepath', type=str, default=None)
    parser.add_argument('--params_filepath', type=str, default=None, help='json file specifying hyperparameters')
    args = parser.parse_args()
    return args

def get_pretrained_model(embedding_dim,  load_filepath:str, temperature=1.0, input_shape=(None, 32, 32, 3)) -> tf.keras.Model:
    
    model = get_model(embedding_dim, temperature, input_shape)
    model.load_weights(load_filepath)
    return model

def get_dataset():
    return get_supervised_dataset()

def run_training(params, args):

    (x_train, y_train), (x_test, y_test) = get_dataset()
    
    pretrained_model = get_pretrained_model(
        params['embedding_dim'],
        args.load_filepath,
        params['temperature'],
    )

    encoder = pretrained_model.encoder
    encoder = tf.keras.Sequential(
        encoder.layers[:-3]
    )
    #print(encoder.summary())
    viewmaker = pretrained_model.viewmaker
    model = get_transfer_classifier(viewmaker, encoder, 10)

    optimizer = tfa.optimizers.SGDW(
         learning_rate = params['learning_rate'],
         momentum = params['momentum'],
         weight_decay = params['weight_decay']
    )
    # optimizer = tfa.optimizers.AdamW(params['weight_decay'])

    model.compile(
        optimizer=optimizer, 
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy']
    )

    model.fit(
        x_train, 
        y_train, 
        validation_data=(x_test, y_test), 
        batch_size=params['batch_size'], 
        epochs=params['epochs']
    )

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