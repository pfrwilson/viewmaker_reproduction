import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_lattice as tfl

ACTIVATIONS = {
    'relu': tf.keras.layers.ReLU,
    'leaky_relu': tf.keras.layers.LeakyReLU,
}

class Viewmaker(tf.keras.Model):
    def __init__(self, num_channels=3, distortion_budget=0.05, activation='relu',
                clamp=True, frequency_domain=False, downsample_to=False, num_res_blocks=3):

        '''Initialize the Viewmaker network.
        Args:
            num_channels: Number of channels in the input (e.g. 1 for speech, 3 for images)
                Input will have shape [batch_size, num_channels, height, width]
            distortion_budget: Distortion budget of the viewmaker (epsilon, in the paper).
                Controls how strong the perturbations can be.
            activation: The activation function used in the network ('relu' and 'leaky_relu' currently supported)
            clamp: Whether to clamp the outputs to [0, 1] (useful to ensure output is, e.g., a valid image)
            frequency_domain: Whether to apply perturbation (and distortion budget) in the frequency domain.
                This is useful for shifting the inductive bias of the viewmaker towards more global / textural views.
            downsample_to: Downsamples the image, applies viewmaker, then upsamples. Possibly useful for 
                higher-resolution inputs, but not evaluaed in the paper.
            num_res_blocks: Number of residual blocks to use in the network.
        '''
        
        super().__init__()
        
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.activation = activation
        self.clamp = clamp
        self.frequency_domain = frequency_domain
        self.downsample_to = downsample_to 
        self.distortion_budget = distortion_budget
        self.act = ACTIVATIONS[activation]

        # Initial convolution layers (+ 1 for noise filter)
        self.conv1 = ConvLayer(self.num_channels + 1, 32, kernel_size=9, stride=1)
        self.in1 = tfa.layers.InstanceNormalization() # may need to add input_spec
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = tfa.layers.InstanceNormalization() # may need to add input_spec
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = tfa.layers.InstanceNormalization() # may need to add input_spec
        
        # Residual layers have +N for added random channels
        self.res1 = ResidualBlock(128 + 1)
        self.res2 = ResidualBlock(128 + 2)
        self.res3 = ResidualBlock(128 + 3)
        self.res4 = ResidualBlock(128 + 4)
        self.res5 = ResidualBlock(128 + 5)

        # Upsampling layers
        self.deconv1 = UpsampleConvLayer(128 + self.num_res_blocks, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = tfa.layers.InstanceNormalization()
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = tfa.layers.InstanceNormalization()
        self.deconv3 = self.deconv3 = ConvLayer(32, self.num_channels, kernel_size=9, stride=1)

    @staticmethod
    def zero_init(m):
        if isinstance(m, (tfl.layers.Linear, tf.keras.layers.Conv2D)):
            # actual 0 has symmetry problems
            m.weights = tf.random.normal(m.weights, mean=0, stddev=1e-4)
            m.bias = tf.convert_to_tensor(np.zeros(shape(m.bias)))
        # elif isinstance(m, tf.keras.layers.BatchNormalization):
        #     pass
            
    def add_noise_channel(self, x, num=1, bound_multiplier=1):
        # bound_multiplier is a scalar or a 1D tensor of length batch_size
        batch_size = x.size(0)
        filter_size = x.size(-1)
        shp = (batch_size, num, filter_size, filter_size)
        bound_multiplier = tf.convert_to_tensor(bound_multiplier) # removed `device` parameter
        noise = tf.random.uniform(shp) * bound_multiplier.view(-1, 1, 1, 1) # removed `device` parameter
        return tf.concatenate((x, noise), axis=1)

    def basic_net(self, y, num_res_blocks=5, bound_multiplier=1):
        if num_res_blocks not in list(range(6)):
            raise ValueError(f'num_res_blocks must be in {list(range(6))}, got {num_res_blocks}.')
        y = self.add_noise_channel(y, bound_multiplier=bound_multiplier)
        y = self.act(self.in1(self.conv1(y)))
        y = self.act(self.in2(self.conv2(y)))
        y = self.act(self.in3(self.conv3(y)))
        # Features that could be useful for other auxilary layers / losses.
        # [batch_size, 128]
        features = y.clone().mean([-1, -2])
    
        for i, res in enumerate([self.res1, self.res2, self.res3, self.res4, self.res5]):
            if i < num_res_blocks:
                y = res(self.add_noise_channel(y, bound_multiplier=bound_multiplier))
        y = self.act(self.in4(self.deconv1(y)))
        y = self.act(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y, features
        
    def get_delta(self, y_pixels, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere'''
        distortion_budget = self.distortion_budget
        delta = tf.keras.activations.tanh(y_pixels) # Project to [-1, 1]
        avg_magnitude = delta.abs().mean([1,2,3], keepdim=True)
        max_magnitude = distortion_budget
        delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta
    
    def call(self, x):
        if self.downsample_to:
            # Downsample.
            x_orig = x
            x = tfa.image.interpolate_bilinear(x)
        y = x
        
        # if self.frequency_domain:
        #     # Input to viewmaker is in frequency domain, outputs frequency domain perturbation.
        #     # Uses the Discrete Cosine Transform.
        #     # shape still [batch_size, C, W, H]
        #     y = tf.signal.dct(y, type=2)

        y_pixels, features = self.basic_net(y, self.num_res_blocks, bound_multiplier=1)
        delta = self.get_delta(y_pixels)
        # if self.frequency_domain:
        #     # Compute inverse DCT from frequency domain to time domain.
        #     delta = tf.signal.idct(delta, type=2)
        if self.downsample_to:
            # Upsample.
            x = x_orig
            delta = tfa.image.interpolate_bilinear(delta)

        # Additive perturbation
        result = x + delta
        if self.clamp:
            result = tf.clip_by_value(result, 0, 1.0)

        return result

class ConvLayer(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = tf.keras.layers.ZeroPadding2D(reflection_padding)
        self.conv2d = tf.keras.layers.Conv2D(out_channels,
                                            kernel_size, 
                                            stride, 
                                            data_format='channels_first')

    def call(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

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

    def call(self, x):
        residual = x
        out = self.act(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

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
        self.reflection_pad = tf.keras.layers.ZeroPadding2D(reflection_padding)
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