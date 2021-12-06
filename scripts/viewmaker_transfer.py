import tensorflow as tf
from tensorflow_addons.optimizers import SGDW
import hydra
import os
import shutil
from omegaconf import DictConfig

from src.datasets.data_loader_factory import get_data_loader

from src.models.resnet_small import ResNet18
from src.models.layers import Identity
from src.models.SimCLR import SimCLR_adversarial
from src.models.viewmaker_new import Viewmaker
from src.models.transfer_learning import TransferModel

os.chdir('..')
CONFIG_PATH: str = os.path.join(os.getcwd(), 'configs')
CONFIG_NAME: str = 'viewmaker_expt_config.yaml'


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(args: DictConfig) -> None:
    # ==========================
    # Setup Saving and logging
    # ==========================

    expt_dir = args.experiment_directory
    if not os.path.isdir(expt_dir):
        os.mkdir(expt_dir)

    shutil.copyfile(
        src=os.path.join(CONFIG_PATH, CONFIG_NAME),
        dst=os.path.join(expt_dir, 'expt_configs.yaml')
    )

    if not os.path.isdir(os.path.join(expt_dir, 'pretrain_log')):
        os.mkdir(os.path.join(expt_dir, 'pretrain_log'))
    if not os.path.isdir(os.path.join(expt_dir, 'transfer_log')):
        os.mkdir(os.path.join(expt_dir, 'transfer_log'))

    # ==========================
    # Dataset
    # ==========================
    dataloader = get_data_loader(args.data.dataset_name)

    (x_train, y_train), (x_test, y_test) = dataloader.get_dataset()
    input_shape = x_train.shape[1:]
    input_shape = (None, *input_shape)  # include None for batch dimension
    num_classes = y_train.shape[1]

    # =======================
    # Build transfer model
    # =======================

    encoder = ResNet18(
        input_shape=input_shape,
        classes=num_classes
    )
    encoder.fc = Identity()  # drop last fully connected layer
    encoder.build(input_shape=input_shape)
    encoder.load_weights(
        os.path.join(expt_dir, 'encoder_weights.h5')
    )
    encoder.pool = Identity()  # encoder output is pre-pool 4x4x512 feature map

    preprocessing = dataloader.get_preprocessing_layer()

    viewmaker = Viewmaker(
        num_channels=input_shape[-1],
        distortion_budget=args.model.distortion_budget
    )
    viewmaker.build(input_shape=input_shape)
    viewmaker.load_weights(
        os.path.join(expt_dir, 'viewmaker_weights.h5')
    )

    encoder.pool = Identity()  # encoder output is pre-pool 4x4x512 feature map

    classifier = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes),
        tf.keras.layers.Softmax()
    ])

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

    optimizer = SGDW(
        weight_decay=args.transfer.weight_decay,
        learning_rate=args.transfer.learning_rate,
        momentum=args.transfer.momentum
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
        batch_size=args.transfer.batch_size,
        epochs=args.transfer.epochs,
        callbacks=[tensorboard_callback],
        validation_split=0.2
    )

    transfer_model.save_weights(
        os.path.join(expt_dir, 'transfer_model_weights.h5')
    )

    # ==============================
    # Evaluation
    # ==============================

    eval_dict = transfer_model.evaluate(
        x_test,
        y_test,
        return_dict=True
    )

    with open(os.path.join(expt_dir, 'evaluation_log.txt'), 'w') as log:
        for key, value in eval_dict.items():
            log.write(f'{key}: {value}')


if __name__ == '__main__':
    main()
