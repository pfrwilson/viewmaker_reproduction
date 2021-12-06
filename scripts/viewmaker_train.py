
import tensorflow as tf
import tensorflow_addons as tfa
import hydra
import os, shutil
from omegaconf import DictConfig

from src.datasets.data_loading import get_data_loader

from src.models.resnet_small import ResNet18
from src.models.layers import Identity
from src.models.SimCLR import SimCLR_adversarial
from src.models.viewmaker_new import Viewmaker
from src.models.transfer_learning import TransferModel

CONFIG_PATH:str = '../configs'
CONFIG_NAME:str = 'viewmaker_expt_config.yaml'

@hydra.main(config_path = CONFIG_PATH, config_name = CONFIG_NAME)
def main(args: DictConfig) -> None:

    # ==========================
    # Setup Saving and logging
    # ==========================
    
    expt_dir = args.experiment_directory
    if not os.isdir(expt_dir):
        os.mkdir(expt_dir)
    
    shutil.copyfile(
        src = os.path.join(CONFIG_PATH, CONFIG_NAME), 
        dst = os.path.join(expt_dir, 'expt_configs.yaml')
    )

    os.mkdir(os.path.join(expt_dir, 'pretrain_log'))
    os.mkdir(os.path.join(expt_dir, 'transfer_log'))

    # ==========================
    # Dataset
    # ==========================    
    dataloader = get_data_loader(args.dataset_name)

    (x_train, y_train), (x_test, y_test) = dataloader.get_dataset()
    input_shape = x_train.shape[1:]
    num_classes = y_train.shape[1:]

    # ==========================
    # Pretrainable Model
    # ==========================

    encoder = ResNet18(
        input_shape = input_shape,
        classes=num_classes
    )
    encoder.fc = Identity()         # drop last fully connected layer

    projection_head = tf.keras.layers.Dense(args.model.embedding_dim)

    preprocessing_layer = dataloader.get_preprocessing_layer()

    viewmaker = Viewmaker(
        num_channels = input_shape[-1],
        distortion_budget = args.model.distortion_budget
    )

    model = SimCLR_adversarial(
        encoder, 
        preprocessing_layer, 
        viewmaker, 
        projection_head,
        temperature = args.model.temperature, 
        viewmaker_loss_weight = args.model.viewmaker_loss_weight
    )

    model.build(input_shape=input_shape)

    # ========================
    # Unsupervised Pretraining
    # ========================

    optimizer = tfa.optimizers.SGDW(
        learning_rate = args.pretrain.learning_rate,
        weight_decay = args.pretrain.weight_decay, 
        momentum = args.pretrain.momentum
    )
    
    model.compile(optimizer=optimizer)

    # log pretraining
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        os.path.join(expt_dir, 'pretrain_log')
    )

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.batch(args.pretrain.batch_size)
    dataset = dataset.take(len(dataset)-1)

    model.fit(dataset, epochs=args.epochs, callbacks=[tensorboard_callback])

    model.save_weights(
        os.path.join(expt_dir, 'pretrained_model_weights.h5')
    )

    # =======================
    # Build transfer model
    # =======================

    encoder = model.encoder
    encoder.pool = Identity()               # encoder output is pre-pool 4x4x512 feature map

    preprocessing = model.preprocessing_layer

    classifier = tf.keras.Sequential([
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(num_classes),
        tf.keras.layers.Softmax()
    ])

    viewmaker = model.viewmaker

    transfer_model = TransferModel(
        encoder, 
        classifier, 
        preprocessing,
        viewmaker=viewmaker
    )

    transfer_model.build(input_shape=input_shape)

    # ============================
    # Finetuning
    # ============================

    optimizer = tfa.optimizers.SGDW(
        weight_decay = args.transfer.weight_decay,
        learning_rate = args.transfer.learning_rate,
        momentum = args.transfer.momentum
    )

    transfer_model.compile(
        optimizer, 
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(expt_dir, 'transfer_log')
    )

    transfer_model.fit(
        x=x_train, 
        y=y_train, 
        batch_size = args.transfer.batch_size,
        epochs=args.epochs,
        callbacks=[tensorboard_callback],
        validation_split=0.2
    )

    model.save_weights(
        os.path.join(expt_dir, 'transfer_model_weights.h5')
    )

    # ==============================
    # Evaluation
    # ==============================

    eval_dict = transfer_model.evaluate(
        x_test, 
        y_test, 
        return_dict = True
    )

    with open(os.path.join(expt_dir, 'evaluation_log.txt'), 'w') as log:
        for key, value in eval_dict.items:
            log.write(f'{key}: {value}')

    
if __name__ == '__main__':
    main()


