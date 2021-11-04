import tensorflow as tf

class Identity(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    def call(self, x):
        return x