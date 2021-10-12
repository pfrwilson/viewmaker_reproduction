import tensorflow as tf
import tensorflow_addons as tfa
from conv_layer import ConvLayer

ACTIVATIONS = {
    'relu': tf.nn.ReLU,
    'leaky_relu': tf.nn.leaky_relu,
}

class ResidualBlock(tf.keras.Model):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, activation='relu'):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = tfa.layers.InstanceNormalization()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = tfa.layers.InstanceNormalization()
        self.act = ACTIVATIONS[activation]()

    def forward(self, x):
        residual = x
        out = self.act(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out