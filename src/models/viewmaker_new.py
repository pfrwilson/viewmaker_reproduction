
import tensorflow as tf
from src.models.layers import Identity
import tensorflow_addons as tfa



class RandomNoise(tf.keras.layers.Layer):
    """concatenates the input with a layer of random noise along channel axis"""
    
    def __init__(self, num=1):
        super(RandomNoise, self).__init__()
        self.num = 1
    
    def call(self, x):
        shape = tf.shape(x)
        noise = tf.random.uniform(shape)
        noise = noise[:, :, :, :1]
        x = tf.concat([x, noise], axis=-1)
        return x


class ResidualBlock(tf.keras.layers.Layer):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=1, padding='same')
        self.in1 = tfa.layers.InstanceNormalization()
        self.conv2 = tf.keras.layers.Conv2D(channels, kernel_size=3, strides=1, padding='same')
        self.in2 = tfa.layers.InstanceNormalization()
        self.act = tf.keras.layers.ReLU()

    def call(self, x):
        residual = x
        out = self.act(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(tf.keras.layers.Layer):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, out_channels, kernel_size, strides, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        
        self.upsample = Identity()
        if upsample:
            self.upsample = tf.keras.layers.UpSampling2D(
                upsample,
                interpolation='nearest'
            )
        self.conv2d = tf.keras.layers.Conv2D(
            out_channels, 
            kernel_size, 
            strides=strides, 
            padding='same'
        )

    def call(self, x):
        x = self.upsample(x)
        x = self.conv2d(x)
        return x


class ProjectToL1Sphere(tf.keras.layers.Layer):
    """Projects the input to an L1 sphere.
    
    A layer which projects its input onto an L1 sphere by passing through tanh and scaling.
    This is used in the viewmaker network to constrain the size of the perturbation it can 
    add to the original image. 

    -------
    attributes: 
        "distortion_budget" : radius of l1 sphere onto which input is projected
        "eps" : small constant to prevent divide by zero errors
    """

    def __init__(self, distortion_budget, eps=1e-4):
        super().__init__()
        self.distortion_budget = distortion_budget
        self.eps = eps
    
    def call(self, x):
        delta = tf.keras.activations.tanh(x) # Project to [-1, 1]
        avg_magnitude = tf.math.reduce_mean(tf.math.abs(delta), [1,2,3], keepdims=True)
        delta = delta * self.distortion_budget / (avg_magnitude + self.eps)
        return delta


class Viewmaker(tf.keras.Model):
    def __init__(self, num_channels=3, distortion_budget=0.05,
                clamp=False, frequency_domain=False):

        '''Initialize the Viewmaker network.
        Args:
            num_channels: Number of channels in the input (e.g. 1 for speech, 3 for images)
                Input will have shape [batch_size, height, width, num_channels]
            distortion_budget: Distortion budget of the viewmaker (epsilon, in the paper).
                Controls how strong the perturbations can be.
            activation: The activation function used in the network ('relu' and 'leaky_relu' currently supported)
            clamp: Whether to clamp the outputs to [0, 1] (useful to ensure output is, e.g., a valid image)
            frequency_domain: Whether to apply perturbation (and distortion budget) in the frequency domain.
                This is useful for shifting the inductive bias of the viewmaker towards more global / textural views.
            downsample_to: Downsamples the image, applies viewmaker, then upsamples. Possibly useful for 
                higher-resolution inputs, but not evaluaed in the paper.
            num_res_blocks: Number of residual blocks to use in the network.
                ---> reproduction - we only used 3 blocks
        '''
        super().__init__()
        self.num_channels = num_channels
        self.clamp = clamp
        self.frequency_domain = frequency_domain
        self.distortion_budget = distortion_budget
        
        self.act = tf.keras.activations.relu
        self.add_noise = RandomNoise()

        # Initial convolution layers (+ 1 for noise filter)
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=9, strides=1, padding='same')
        self.in1 = tfa.layers.InstanceNormalization() # may need to add input_spec
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same')
        self.in2 = tfa.layers.InstanceNormalization() # may need to add input_spec
        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same')
        self.in3 = tfa.layers.InstanceNormalization() # may need to add input_spec
        
        # Residual layers have +N for added random channels
        self.res1 = ResidualBlock(128 + 1)
        self.res2 = ResidualBlock(128 + 2)
        self.res3 = ResidualBlock(128 + 3)

        # Upsampling layers
        self.deconv1 = UpsampleConvLayer(64, kernel_size=3, strides=1, upsample=2)
        self.in4 = tfa.layers.InstanceNormalization()
        self.deconv2 = UpsampleConvLayer(32, kernel_size=3, strides=1, upsample=2)
        self.in5 = tfa.layers.InstanceNormalization()
        self.deconv3 = tf.keras.layers.Conv2D(self.num_channels, kernel_size=9, strides=1, padding='same')   

        # Project to sphere layer
        self.project = ProjectToL1Sphere(self.distortion_budget)


    def call(self, x):
        y = x
        y = self.add_noise(y)

        y = self.act(self.in1(self.conv1(y)))
        y = self.act(self.in2(self.conv2(y)))
        y = self.act(self.in3(self.conv3(y)))

        y = self.res1(self.add_noise(y))
        y = self.res2(self.add_noise(y))
        y = self.res3(self.add_noise(y))

        y = self.act(self.in4(self.deconv1(y)))
        y = self.act(self.in5(self.deconv2(y)))
        y = self.deconv3(y)

        delta = self.project(y)
        x = x + delta

        return x