"""Partitioned Embedding Layer"""

import logging
from typing import Tuple

import tensorflow as tf

from deepr.layers import base
from deepr.layers.core import Dense


LOGGER = logging.getLogger(__name__)


class Embedding(base.Layer):
    """Partitioned Embedding Layer"""

    def __init__(
        self,
        variable_name: str,
        shape: Tuple[int, ...],
        trainable: bool = True,
        initializer=None,
        num_shards: int = None,
        reuse: bool = None,
        partition_strategy: str = "div",
        **kwargs,
    ):
        super().__init__(n_in=1, n_out=1, **kwargs)
        self.variable_name = variable_name
        self.trainable = trainable
        self.shape = shape
        self.num_shards = num_shards
        self.initializer = initializer
        self.reuse = reuse
        self.partition_strategy = partition_strategy

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=self.reuse):
            embeddings_var = tf.compat.v1.get_variable(
                name=self.variable_name,
                trainable=self.trainable,
                shape=self.shape,
                initializer=self.initializer,
                partitioner=tf.compat.v1.fixed_size_partitioner(num_shards=self.num_shards, axis=0)
                if self.num_shards is not None
                else None,
            )
            return tf.nn.embedding_lookup(
                params=embeddings_var, ids=tf.maximum(tensors, 0), partition_strategy=self.partition_strategy
            )


@base.layer(n_in=2, n_out=1)
def CombineEmbeddings(tensors, mode, output_dim, project=True):
    """Combine Embeddings Layers"""
    embedding = tf.concat(tensors, axis=-1)
    if project:
        embedding = Dense(units=output_dim)(embedding, mode=mode)
    return embedding
