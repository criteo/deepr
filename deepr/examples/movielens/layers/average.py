# pylint: disable=unexpected-keyword-arg,no-value-for-parameter
"""Average Model."""

import tensorflow as tf
import deepr as dpr


def AverageModel(vocab_size: int, dim: int):
    """Average Model."""
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("inputPositives", "inputMask")),
        dpr.layers.Embedding(
            inputs="inputPositives", outputs="inputEmbeddings", variable_name="embeddings", shape=(vocab_size, dim)
        ),
        dpr.layers.ToFloat(inputs="inputMask", outputs="inputWeights"),
        dpr.layers.WeightedAverage(inputs=("inputEmbeddings", "inputWeights"), outputs="userEmbeddings"),
        Logits(inputs="userEmbeddings", outputs="logits"),
        dpr.layers.Select(inputs=("userEmbeddings", "logits")),
    )


@dpr.layers.layer(n_in=1, n_out=1)
def Logits(tensors: tf.Tensor, vocab_size: int, dim: int):
    embeddings = tf.get_variable(name="embeddings", shape=(vocab_size, dim), reuse=True)
    biases = tf.get_variable(name="biases", shape=(vocab_size,), initializer=tf.zeros_initializer())
    logits = tf.linalg.matmul(tensors, embeddings, transpose_b=True) + tf.expand_dims(biases, axis=0)
    return logits
