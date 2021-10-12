import tensorflow as tf

class ConvLayer(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = tf.pad(reflection_padding)
        self.conv2d = tf.keras.layers.Conv2D(out_channels,
                                            kernel_size, 
                                            stride, 
                                            data_format='channels_first')

    def call(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
