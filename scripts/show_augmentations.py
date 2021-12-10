from src.datasets.data_loader_factory import get_data_loader
from src.models.viewmaker_new import Viewmaker
from src.models.layers import Identity

DATASET_NAME = 'cifar_10'
MODE = 'viewmaker'
VIEWMAKER_WEIGHTS_FP = None # r'C:\Users\Paul\viewmaker_reproduction\experiments\test_expt\viewmaker_weights.h5'


def main():
    dataloader = get_data_loader(DATASET_NAME)
    x_train = dataloader.get_dataset_for_pretraining()

    input_shape = dataloader.get_input_shape()

    viewmaker = Identity()
    if MODE == 'viewmaker':
        viewmaker = Viewmaker()
        viewmaker.build(input_shape=(None, *input_shape))
        if VIEWMAKER_WEIGHTS_FP:
            viewmaker.load_weights(VIEWMAKER_WEIGHTS_FP)
    elif MODE == 'expert':
        viewmaker = dataloader.get_augmentation_layer()

    original = next(iter(x_train.batch(batch_size=4)))
    augmented1 = viewmaker(original)
    augmented2 = viewmaker(original)


    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3)
    for i, row in enumerate(axes):
        row[0].imshow(original[i])
        row[1].imshow(augmented1[i])
        row[2].imshow(augmented2[i])
        row[0].set_xticks([])
        row[0].set_yticks([])
        row[1].set_xticks([])
        row[1].set_yticks([])
        row[2].set_xticks([])
        row[2].set_yticks([])

    fig.set_figheight(12)
    fig.set_figwidth(9)
    fig.show()


if __name__ == '__main__':
    main()

