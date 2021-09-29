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

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def call(self, x):
        
        #normalize to unit vectors
        x = x/tf.norm(
            x, ord='euclidean', axis=1, keepdims=True
        )
        
        # x.shape = (2 * batch_size, embedding_dimension)
        batch_size = x.shape[0]//2

        embeddings_1 = x[:batch_size]
        embeddings_2 = x[batch_size:]

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

        loss = tf.reduce_mean(tf.keras.layers.Add()([losses_1, losses_2]))
        
        return loss
