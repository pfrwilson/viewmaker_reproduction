import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_lattice as tfl


ACTIVATIONS = {
    'relu': tf.nn.ReLU,
    'leaky_relu': tf.nn.leaky_relu,
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
        self.act = ACTIVATIONS[activation]()

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
                tf.random.normal(m.weights, mean=0, stddev=1e-4)
                # init.constant_(m.weight.data, 0)
                # TODO: determine how to initalize biases to 0. afaik these are not cleanly separable.
                # init.constant_(m.bias.data, 0)
            elif isinstance(m, tf.keras.layers.BatchNormalization):
                pass

        def add_noise_channel(self, x, num=1, bound_multiplier=1):
            # bound_multiplier is a scalar or a 1D tensor of length batch_size
            batch_size = x.size(0)
            filter_size = x.size(-1)
            shp = (batch_size, num, filter_size, filter_size)
            bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
            noise = torch.rand(shp, device=x.device) * bound_multiplier.view(-1, 1, 1, 1)
            return torch.cat((x, noise), dim=1)