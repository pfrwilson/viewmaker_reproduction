import tensorflow as tf
import math
from tensorflow.python.ops.linalg_ops import norm 

class SimCLR(tf.keras.Model):

    def __init__(
            self, 
            encoder, 
            viewmaker, # a non-trainable viewmaker layer
            projection_head, 
            temperature=1.0, 
            normalize=True, 
            **kwargs
        ):
        super(SimCLR, self).__init__(**kwargs)
        self.encoder = encoder   
        self.projection_head = projection_head    
        self.viewmaker = viewmaker    
        self.loss_fn = SimCLRObjective(temperature=temperature, normalize=normalize)
    
    def compile(self, optimizer):
        super(SimCLR, self).compile()
        self.optimizer = optimizer

    def call(self, x):
      
        x = tf.concat([x, x], axis=0)
        x = self.viewmaker(x)
        x = self.encoder(x)
        x = self.projection_head(x)
        
        return x

    def train_step(self, x):        

        with tf.GradientTape() as tape:
            embeddings = self(x)
            loss = self.loss_fn(embeddings)

        # apply the optimization step
        weights = self.encoder.trainable_weights
        weights.extend(self.projection_head.trainable_weights)
        grads = tape.gradient(loss, weights)
        self.optimizer.apply_gradients(zip(grads, weights))

        return {'SimCLR_loss': loss}


class SimCLR_adversarial(SimCLR):
    
    def __init__(
            self, 
            encoder, 
            viewmaker, 
            projection_head,
            temperature=1.0, 
            normalize=True,
            viewmaker_loss_weight=1.0
            ):
        super(SimCLR_adversarial, self).__init__(
            encoder, 
            viewmaker, 
            projection_head,
            temperature,
            normalize
        )
        self.viewmaker_loss_weight = viewmaker_loss_weight

    def train_step(self, x):

        with tf.GradientTape(persistent=True) as grad:
            
            embeddings = self(x)
            loss = self.loss_fn(embeddings)
            viewmaker_loss = -loss*self.viewmaker_loss_weight

        # update weights
        encoder_weights = self.encoder.trainable_weights
        encoder_grads = grad.gradient(loss, encoder_weights)
        projection_head_weights = self.projection_head.trainable_weights
        projection_head_grads = grad.gradient(loss, projection_head_weights)
        viewmaker_weights = self.viewmaker.trainable_weights
        viewmaker_grads = grad.gradient(viewmaker_loss, viewmaker_weights)

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


class SimCLRObjective(tf.keras.layers.Layer):
    """ Viewmaker Networks Official Implementation of SimCLR loss - slightly different 
    from original"""
    def __init__(self, temperature=1.0, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def call(self, x):
        # x shape (batch_size 2), embedding_dim

        if self.normalize:
            x = tf.math.l2_normalize(x, -1)

        embeddings_1, embeddings_2 = tf.split(x, 2, axis=0)
        batch_size = tf.shape(embeddings_1)[0]

        witness_score = tf.reduce_sum(tf.math.multiply(embeddings_1, embeddings_2), axis=1)/self.temperature

        witness_norm = tf.matmul(embeddings_1, x, transpose_b=True)/self.temperature
        witness_norm = tf.math.reduce_logsumexp(witness_norm, axis=1) 
        # 2 * batch_size as compatible tensor
        NUMBER_OF_EXAMPLES = tf.cast(tf.constant(2)*batch_size, 'float32')
        witness_norm = witness_norm - tf.math.log(NUMBER_OF_EXAMPLES)

        loss = -tf.reduce_mean(witness_score - witness_norm)

        return loss
