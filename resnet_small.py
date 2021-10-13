import tensorflow as tf
import tensorflow.keras.layers as layers


class BasicBlock(layers.Layer):
    expansion = 1
    
    def __init__(self, in_planes, planes, strides=1):

        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(planes, (3,3), strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization(epsilon=0.00001, momentum=0.1)
        self.conv2 = layers.Conv2D(planes, (3,3), strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization(epsilon=0.00001, momentum=0.1)

        #shortcut connection (either identity or convolution to match output shape)
        self.shortcut = tf.keras.Sequential()
        if in_planes != planes or strides != 1:
            self.shortcut.add(
                layers.Conv2D(planes, (1, 1), strides=strides, use_bias=False)
            )
            self.shortcut.add(
                layers.BatchNormalization(epsilon=0.00001, momentum=0.1)
            )

    def call(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = layers.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)
        return x


class Bottleneck(layers.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, strides=1):
        super(Bottleneck, self).__init__() 
        self.conv1 = layers.Conv2D(planes, (1, 1), padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization(epsilon=0.00001, momentum=0.1)
        self.conv2 = layers.Conv2D(planes, (3, 3), strides=strides, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization(epsilon=0.00001, momentum=0.1)
        self.conv3 = layers.Conv2D(planes*self.expansion, (1, 1), padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization(epsilon=0.00001, momentum=0.1)

        self.shortcut = tf.keras.Sequential()
        if in_planes != planes*self.expansion or strides != 1:
            self.shortcut.add(
                layers.Conv2D(planes*self.expansion, (1, 1), strides=strides, use_bias=False)
            )
            self.shortcut.add(
                layers.BatchNormalization(epsilon=0.00001, momentum=0.1)
            )

    def call(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = layers.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = layers.ReLU()(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)
        return x 


class ResNet(tf.keras.Model):
    
    def __init__(self, block, num_blocks, num_classes=10, num_channels=3, input_size=32):
        super(ResNet, self).__init__()
        assert input_size in [32, 64]
        self.in_planes = 64
        self.num_channels = num_channels

        self.conv1 = layers.Conv2D(64, (3,3), strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization(epsilon=0.00001, momentum=0.1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], strides=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], strides=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], strides=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], strides=2)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes)

    def _make_layer(self, block, planes, num_blocks, strides):
        # first block in layer has stride 2, rest have stride 1
        strides = [strides] + (num_blocks-1)*[1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return tf.keras.Sequential(layers)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = layers.ReLU()(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = layers.AvgPool2D(pool_size=(4, 4))(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits

def ResNet18(num_classes, num_channels=3, input_size=32):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, num_channels=num_channels, 
                  input_size=input_size)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)


def ResNet152(num_classes):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)