# pylint: disable=unexpected-keyword-arg,no-value-for-parameter
"""Transformer Model."""

import tensorflow as tf
import numpy as np

import deepr as dpr


def TransformerModel(vocab_size: int, dim: int, **kwargs):
    """Transformer Model."""
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("inputPositives", "inputMask")),
        dpr.layers.Embedding(
            inputs="inputPositives",
            outputs="inputEmbeddings",
            variable_name="embeddings",
            shape=(vocab_size, dim),
            initializer=tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(1 / dim)),
        ),
        dpr.layers.Transformer(inputs=("inputEmbeddings", "inputMask"), outputs="userEmbeddings", dim=dim, **kwargs),
    )
