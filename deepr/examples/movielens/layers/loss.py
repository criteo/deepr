# pylint: disable=unexpected-keyword-arg,no-value-for-parameter
"""BPR Loss with biases."""

import tensorflow as tf

import deepr as dpr


def BPRLoss(vocab_size: int, dim: int):
    """BPR Loss with biases."""
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("userEmbeddings", "targetPositives", "targetNegatives", "targetMask")),
        dpr.layers.Embedding(
            inputs="targetPositives",
            outputs="targetPositiveEmbeddings",
            variable_name="embeddings",
            shape=(vocab_size, dim),
            reuse=True,
        ),
        dpr.layers.Embedding(
            inputs="targetNegatives",
            outputs="targetNegativeEmbeddings",
            variable_name="embeddings",
            shape=(vocab_size, dim),
            reuse=True,
        ),
        dpr.layers.Embedding(
            inputs="targetPositives",
            outputs="targetPositiveBiases",
            variable_name="biases",
            shape=(vocab_size,),
            initializer=tf.zeros_initializer(),
        ),
        dpr.layers.Embedding(
            inputs="targetNegatives",
            outputs="targetNegativeBiases",
            variable_name="biases",
            shape=(vocab_size,),
            reuse=True,
        ),
        dpr.layers.DotProduct(inputs=("userEmbeddings", "targetPositiveEmbeddings"), outputs="targetPositiveProduct"),
        dpr.layers.DotProduct(inputs=("userEmbeddings", "targetNegativeEmbeddings"), outputs="targetNegativeProduct"),
        dpr.layers.Add(inputs=("targetPositiveBiases", "targetPositiveProduct"), outputs="targetPositiveLogits"),
        dpr.layers.Add(inputs=("targetNegativeBiases", "targetNegativeProduct"), outputs="targetNegativeLogits"),
        dpr.layers.ToFloat(inputs="targetMask", outputs="targetWeight"),
        dpr.layers.ExpandDims(inputs="targetMask", outputs="targetMask"),
        dpr.layers.MaskedBPR(
            inputs=("targetPositiveLogits", "targetNegativeLogits", "targetMask", "targetWeight"), outputs="loss"
        ),
        dpr.layers.TripletPrecision(
            inputs=("targetPositiveLogits", "targetNegativeLogits", "targetMask", "targetWeight"),
            outputs="triplet_precision",
        ),
        dpr.layers.Select(inputs=("loss", "triplet_precision")),
    )
