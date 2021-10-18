import tensorflow as tf
import tensorflow_lattice as tfl

class TrainClassifier(tf.keras.Model):
    
    def __init__(self, num_classes, encoder):
        super().__init__()
        self.encoder = encoder
        self.reg = tf.keras.layers.Dense(num_classes, 
                                        activation='softmax')
        self.encoder.trainable = False 

    def call(self, x):
        output = self.reg(self.encoder(x))
        return output 
    