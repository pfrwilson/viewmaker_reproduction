import tensorflow as tf 

class SimCLR(tf.keras.Model):

    def __init__(
            self, 
            encoder, 
            preprocessing_layer, # with augmentations
            projection_head, 
            temperature=1.0, 
            normalize=True, 
            **kwargs
        ):
        super(SimCLR, self).__init__(**kwargs)
        self.encoder = encoder   
        self.projection_head = projection_head    
        self.preprocessing_layer = preprocessing_layer    
        self.embeddings_to_logits = SimCLR_logits_from_embeddings(
            temperature=temperature, 
            normalize=normalize
        )
    
    def compile(self, optimizer, loss_fn=tf.keras.losses.CategoricalCrossentropy()):
        super(SimCLR, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def call(self, x):
        # generates views of the data
        data_view1 = self.preprocessing_layer(x, training=True)
        data_view2 = self.preprocessing_layer(x, training=True)
        batch_size = tf.shape(x)[0]

        # concatenate views for input to encoder
        X = tf.concat([data_view1, data_view2], axis=0)

        # pass to encoder and projection head
        representations = self.encoder(X)
        embeddings = self.projection_head(representations)

        #generate logits
        logits = self.embeddings_to_logits(embeddings)

        predictions = tf.keras.layers.Softmax()(logits)

        return predictions

    def train_step(self, x):        

        batch_size = tf.shape(x)[0]

        with tf.GradientTape() as tape:
            predictions = self(x)
            labels = tf.eye(batch_size*2)

            # compute the sim-clr loss with respect to the logits
            loss = self.loss_fn(labels, predictions)

        # apply the optimization step
        weights = self.encoder.trainable_weights
        weights.extend(self.projection_head.trainable_weights)
        grads = tape.gradient(loss, weights)
        self.optimizer.apply_gradients(zip(grads, weights))

        # compute the accuracy
        accuracy = SimCLR._compute_accuracy(labels, predictions)

        return {'SimCLR_loss': loss,
                'SimCLR_accuracy': accuracy}

    @staticmethod
    def _compute_accuracy(labels, predictions):
        predictions = tf.argmax(predictions, axis=1)
        labels = tf.argmax(labels, axis=1)
        n_labels = tf.shape(labels)[0]
        accuracy = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), 'int32'))/(n_labels)
        return accuracy

class SimCLR_adversarial(SimCLR):
    
    def __init__(
            self, 
            encoder, 
            preprocessing_layer, 
            viewmaker, 
            projection_head,
            temperature=1.0, 
            normalize=True,
            viewmaker_loss_weight=1.0
            ):
        super(SimCLR_adversarial, self).__init__(
            encoder, 
            preprocessing_layer,
            projection_head,
            temperature,
            normalize
        )
        self.viewmaker = viewmaker
        self.viewmaker_loss_weight = viewmaker_loss_weight

    def call(self, x):
        # generates views of the data
        data_view1 = self.preprocessing_layer(x)
        data_view2 = self.preprocessing_layer(x)
        batch_size = tf.shape(x)[0]

        # concatenate views for input to encoder
        x = tf.concat([data_view1, data_view2], axis=0)

        # pass to viewmaker, encoder and projection head
        views = self.viewmaker(x)
        representations = self.encoder(views)
        embeddings = self.projection_head(representations)

        #generate logits
        logits = self.embeddings_to_logits(embeddings)

        predictions = tf.keras.layers.Softmax()(logits)

        return predictions

    def train_step(self, x):

        batch_size = tf.shape(x)[0]

        with tf.GradientTape(persistent=True) as grad:
            
            predictions = self(x)
            labels = tf.eye(batch_size*2)
            
            # compute the sim-clr loss with respect to the logits
            loss = self.loss_fn(labels, predictions)
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

        # compute the accuracy
        accuracy = self._compute_accuracy(labels, predictions)

        return {'SimCLR loss': loss,
                'SimCLR accuracy': accuracy}


class SimCLR_logits_from_embeddings(tf.keras.layers.Layer):
    """Receives batch of embeddings and computes similarity logit matrix
    logit matrix can be passed to softmax_cross_entropy loss function
    with labels eye(batch_size) (identity matrix of dim batch_size)"""
    def __init__(self, temperature=1.0, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def call(self, x):
        #normalize to unit vectors
        # Get (normalized) hidden1 and hidden2.
        if self.normalize:
            x = tf.math.l2_normalize(x, -1)

        embeddings_1, embeddings_2 = tf.split(x, 2, 0)
        batch_size = tf.shape(embeddings_1)[0]

        masks = tf.one_hot(tf.range(batch_size), batch_size)

        LARGE_NUM = 1e5
        logits_11 = tf.matmul(embeddings_1, embeddings_1, transpose_b=True)/self.temperature
        logits_11 = tf.subtract(logits_11, LARGE_NUM*masks)
        logits_22 = tf.matmul(embeddings_2, embeddings_2, transpose_b=True)/self.temperature
        logits_22 = tf.subtract(logits_22, LARGE_NUM*masks)
        logits_12 = tf.matmul(embeddings_1, embeddings_2, transpose_b=True)/self.temperature
        logits_21 = tf.matmul(embeddings_2, embeddings_1, transpose_b=True)/self.temperature

        logits_1 = tf.concat([logits_12, logits_11], axis=1)
        logits_2 = tf.concat([logits_22, logits_21], axis=1)

        logits = tf.concat([logits_1, logits_2], axis=0)
        predictions = tf.keras.layers.Softmax()(logits)
        return logits