import tensorflow as tf 
from losses import SimCLR_loss, AdversarialSimCLRLoss

class SimCLR(tf.keras.Model):

    def __init__(self, encoder, augmentation, projection_head):
        super(SimCLR, self).__init__()
        # encoder: callable on input data - returns a high-dimensional representation of its input
        self.encoder = encoder   
        # projection head - callable on the output of the encoding layer - projects high dimensional 
        # data into a lower dimension       
        self.projection_head = projection_head    
        # augmentation layer - callable, layer maps input to "augmented" view of input 
        self.augmentation = augmentation
    
    def compile(self, optimizer, loss_fn = SimCLR_loss()):
        super(SimCLR, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):

        # generate views of the data
        data_view1 = self.augmentation(data)
        data_view2 = self.augmentation(data)

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


class SimCLR_adversarial(tf.keras.Model):
    
    def __init__(self, encoder, viewmaker, projection_head):
        super(SimCLR_adversarial, self).__init__()
        # encoder: callable on input data - returns a high-dimensional representation of its input
        self.encoder = encoder
        # viewmaker: callable on input data - returns a distorted "view" of its input
        self.viewmaker = viewmaker
        # projection head - callable on the output of the encoding layer - projects high dimensional 
        # data into a lower dimension
        self.projection_head = projection_head

    def compile(self, optimizer, loss_fn=AdversarialSimCLRLoss()):
        self.encoder_optimizer = optimizer.copy()
        self
        # loss_fn must return (loss, adversarial_loss)
        self.loss_fn = loss_fn

    def train_step(self, data):

        with tf.GradientTape(persistent=True) as grad:
            
            # generate views of the data
            data_view1 = self.viewmaker(data)
            data_view2 = self.viewmaker(data)
            
            # concatenate views for input to encoder
            X = tf.concat([data_view1, data_view2], axis=0)

            # pass X to encoder and projection head
            representations = self.encoder(X)

            embeddings = self.projection_head(representations)
            
            # compute the sim-clr loss with respect to the embeddings
            loss, adversarial_loss = self.loss_fn(embeddings)

        # update weights
        encoder_weights = self.encoder.trainable_weights
        encoder_grads = grad.gradient(loss, encoder_weights)
        projection_head_weights = self.projection_head.trainable_weights
        projection_head_grads = grad.gradient(loss, projection_head_weights)
        viewmaker_weights = self.viewmaker.trainable_weights
        viewmaker_grads = grad.gradient(adversarial_loss, viewmaker_weights)

        weights = []
        weights.extend(encoder_weights)
        weights.extend(projection_head_weights)
        weights.extend(viewmaker_weights)

        grads = []
        grads.extend(encoder_grads)
        grads.extend(projection_head_grads)
        grads.extend(viewmaker_grads)

        self.optimizer.apply_gradients(zip(grads, weights))

        return {'SimCLR loss': loss}