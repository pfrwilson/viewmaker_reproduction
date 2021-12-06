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

    # ==========================
    # Pretrainable Model
    # ==========================

    encoder = ResNet18(
        input_shape=input_shape,
        classes=num_classes
    )
    encoder.fc = Identity()  # drop last fully connected layer

    projection_head = tf.keras.layers.Dense(args.model.embedding_dim)

    preprocessing_layer = dataloader.get_preprocessing_layer()

    viewmaker = Viewmaker(
        num_channels=input_shape[-1],
        distortion_budget=args.model.distortion_budget
    )

    model = SimCLR_adversarial(
        encoder,
        preprocessing_layer,
        viewmaker,
        projection_head,
        temperature=args.model.temperature,
        viewmaker_loss_weight=args.model.viewmaker_loss_weight
    )

    model.build(input_shape=input_shape)

    # ========================
    # Unsupervised Pretraining
    # ========================

    optimizer = SGDW(
        learning_rate=args.pretrain.learning_rate,
        weight_decay=args.pretrain.weight_decay,
        momentum=args.pretrain.momentum
    )

    model.compile(optimizer=optimizer)

    # log pretraining
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        os.path.join(expt_dir, 'pretrain_log')
    )

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.batch(args.pretrain.batch_size)
    dataset = dataset.take(len(dataset) - 1)

    model.fit(dataset, epochs=args.pretrain.epochs, callbacks=[tensorboard_callback])

    # ========================
    # Saving the weights
    # ========================

    model.encoder.save_weights(
        os.path.join(expt_dir, 'encoder_weights.h5')
    )

    model.viewmaker.save_weights(
        os.path.join(expt_dir, 'viewmaker_weights.h5')
    )


if __name__ == '__main__':
    main()

