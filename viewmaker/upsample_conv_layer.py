import tensorflow as tf

class UpsampleConvLayer(tf.keras.Model):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = tf.pad(reflection_padding)
        self.conv2d = tf.keras.layers.Conv2D(out_channels, 
                                            kernel_size, 
                                            stride,
                                            data_format='channels_first')

    def call(self, x):
        x_in = x
        if self.upsample:
            x_in = tf.image.resize(x_in, method=ResizeMethod.NEAREST_NEIGHBOR)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out