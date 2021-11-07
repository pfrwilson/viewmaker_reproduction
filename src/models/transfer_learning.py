import tensorflow as tf

class TransferModel(tf.keras.Model):
    def __init__(
        self, 
        encoder,
        classifier,
        preprocessing,
        viewmaker = None, 
        freeze_encoder_weights = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.preprocessing = preprocessing
        self.viewmaker = viewmaker
        if self.viewmaker:
            self.viewmaker.trainable=False
        self.encoder = encoder
        if freeze_encoder_weights:
            self.encoder.trainable=False
        self.classifier = classifier               
    
    def call(self, x, training=False):
        x = self.preprocessing(x, training=training)
        if training and self.viewmaker:
            x = self.viewmaker(x)
        x = self.encoder(x)
        predictions = self.classifier(x)
        return predictions