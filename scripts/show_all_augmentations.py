from src.datasets.data_loader_factory import get_data_loader
from src.models.viewmaker_new import Viewmaker
from src.models.layers import Identity

DATASET_NAME = 'cifar_10'
MODE = 'viewmaker'
VIEWMAKER_WEIGHTS_FP = r'C:\Users\Paul\viewmaker_reproduction\experiments\viewmaker_weights.h5'


def main():
    dataloader = get_data_loader(DATASET_NAME)
    x_train = dataloader.get_dataset_for_pretraining()

    input_shape = dataloader.get_input_shape()

    viewmaker = Viewmaker()
    viewmaker.build(input_shape=(None, *input_shape))
    if VIEWMAKER_WEIGHTS_FP:
        viewmaker.load_weights(VIEWMAKER_WEIGHTS_FP)
    augmenter = dataloader.get_augmentation_layer()

    original = next(iter(x_train.batch(batch_size=4)))
    augmented1 = augmenter(original)
    augmented2 = augmenter(original)
    viewmaker1 = viewmaker(original)
    viewmaker2 = viewmaker(original)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 5)
    for i, row in enumerate(axes):
        for axis in row:
            axis.set_xticks([])
            axis.set_yticks([])
        row[0].imshow(original[i])
        row[1].imshow(augmented1[i])
        row[2].imshow(augmented2[i])
        row[3].imshow(viewmaker1[i])
        row[4].imshow(viewmaker2[i])

    fig.set_figheight(12)
    fig.set_figwidth(18)
    fig.show()


if __name__ == '__main__':
    main()

