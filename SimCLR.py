import tensorflow as tf 

class SimCLR(tf.keras.Model):

    def __init__(self, encoder, augmentation, projection_head):
        super(SimCLR, self).__init__()
        self.encoder = encoder                     
        self.projection_head = projection_head     
        self.augmentation = augmentation
    
    def compile(self, optimizer, loss_fn):
        super(SimCLR, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):

        # generate views of the data
        data_view1 = tf.map_fn(self.augmentation, data)
        data_view2 = tf.map_fn(self.augmentation, data)

        # concatenate views for input to encoder
        X = tf.concat([data_view1, data_view2], axis=0)

        with tf.GradientTape() as tape:
            # pass X to encoder and projection head
            representations = self.encoder(X)

            embeddings = self.projection_head(representations)
            
            # compute the sim-clr loss with respect to the embeddings
            loss = self.loss_fn(embeddings)

        weights = self.encoder.trainable_weights
        weights.extend(self.projection_head.trainable_weights)
        grads = tape.gradient(loss, weights)
        self.optimizer.apply_gradients(zip(grads, weights))

        return {'SimCLR_loss': loss}