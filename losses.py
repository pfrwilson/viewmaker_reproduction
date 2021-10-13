# Adapted from https://github.com/google-research/simclr/blob/master/objective.py
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.

import tensorflow as tf

LARGE_NUM = 1e9


class SimCLR_loss(tf.keras.layers.Layer):

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
        labels = tf.one_hot(tf.range(batch_size), batch_size*2)

        logits_11 = tf.matmul(embeddings_1, embeddings_1, transpose_b=True)/self.temperature
        logits_11 = tf.subtract(logits_11, LARGE_NUM*masks)
        logits_22 = tf.matmul(embeddings_2, embeddings_2, transpose_b=True)/self.temperature
        logits_22 = tf.subtract(logits_22, LARGE_NUM*masks)
        logits_12 = tf.matmul(embeddings_1, embeddings_2, transpose_b=True)/self.temperature
        logits_21 = tf.matmul(embeddings_2, embeddings_1, transpose_b=True)/self.temperature

        losses_1 = tf.nn.softmax_cross_entropy_with_logits(
            labels, tf.concat([logits_12, logits_11], axis=1)
        )
        losses_2 = tf.nn.softmax_cross_entropy_with_logits(
            labels, tf.concat([logits_21, logits_22], axis=1)
        )

        loss = losses_1 + losses_2
        
        return loss

class AdversarialSimCLRLoss(tf.keras.layers.Layer):

    def __init__(self, viewmaker_loss_weight=1.0, temperature=1.0, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.viewmaker_loss_weight=viewmaker_loss_weight
    
    def call(self, x):
        simclr_loss = SimCLR_loss(self.temperature, self.normalize)(x)
        viewmaker_loss = -simclr_loss * self.viewmaker_loss_weight
        return simclr_loss, viewmaker_loss


