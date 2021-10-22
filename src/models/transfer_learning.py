import tensorflow as tf

def get_transfer_classifier(viewmaker, encoder, num_classes):
    return TransferModel(
        viewmaker, 
        encoder, 
        tf.keras.Dense(num_classes),
        )

class TransferModel(tf.keras.Model):
    def __init__(
        self, 
        encoder,
        classifier,
        viewmaker = None, #optional augmentation layer for training
        freeze_encoder_weights = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.viewmaker = viewmaker
        self.encoder = encoder
        if freeze_encoder_weights:
            self.encoder.trainable=False
        self.classifier = classifier
    
    def call(self, x, training=False):
        if training and self.viewmaker:
            x = self.viewmaker(x)
        x = self.encoder(x)
        logits = self.classifier(x)
        return logits