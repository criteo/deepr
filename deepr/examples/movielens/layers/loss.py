# pylint: disable=unexpected-keyword-arg,no-value-for-parameter,invalid-name
"""BPR Loss with biases."""

from typing import Tuple

import tensorflow as tf

import deepr as dpr


def VAELoss(beta: float, **kwargs):
    BPR = BPRLoss(**kwargs)
    inputs = list(BPR.inputs) + ["KL"]
    outputs = list(BPR.outputs) + ["loss_bpr", "loss_KL"]
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=inputs, outputs=inputs),
        BPR,
        dpr.layers.Select(inputs="loss", outputs="loss_bpr"),
        dpr.layers.Select(inputs="KL", outputs="loss_KL"),
        AddKL(inputs=("loss", "KL"), outputs=("loss"), beta=beta),
        dpr.layers.Select(inputs=outputs, outputs=outputs),
    )


@dpr.layers.layer(n_in=2, n_out=1)
def AddKL(tensors: Tuple[tf.Tensor], beta: float):
    loss, KL = tensors
    return loss + beta * KL


def BPRLoss(vocab_size: int, dim: int, reuse_embeddings: bool = True, embeddings_name: str = "embeddings"):
    """BPR Loss with biases."""
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("userEmbeddings", "targetPositives", "targetNegatives", "targetMask")),
        dpr.layers.Embedding(
            inputs="targetPositives",
            outputs="targetPositiveEmbeddings",
            variable_name=embeddings_name,
            shape=(vocab_size, dim),
            reuse=reuse_embeddings,
        ),
        dpr.layers.Embedding(
            inputs="targetNegatives",
            outputs="targetNegativeEmbeddings",
            variable_name=embeddings_name,
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
