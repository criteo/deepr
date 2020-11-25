# pylint: disable=unexpected-keyword-arg,no-value-for-parameter,invalid-name
"""BPR Loss with biases."""

import deepr


def BPRLoss(vocab_size: int):
    """BPR Loss with biases."""
    return deepr.layers.DAG(
        deepr.layers.Select(inputs=("userEmbeddings", "targetPositives", "targetNegatives", "targetMask")),
        deepr.layers.DenseIndex(
            inputs=("userEmbeddings", "targetPositives"),
            outputs="targetPositiveLogits",
            units=vocab_size,
            kernel_name="embeddings",
            bias_name="biases",
            reuse=True,
        ),
        deepr.layers.DenseIndex(
            inputs=("userEmbeddings", "targetNegatives"),
            outputs="targetNegativeLogits",
            units=vocab_size,
            kernel_name="embeddings",
            bias_name="biases",
            reuse=True,
        ),
        deepr.layers.ToFloat(inputs="targetMask", outputs="targetWeight"),
        deepr.layers.ExpandDims(inputs="targetMask", outputs="targetMask"),
        deepr.layers.MaskedBPR(
            inputs=("targetPositiveLogits", "targetNegativeLogits", "targetMask", "targetWeight"), outputs="loss"
        ),
        deepr.layers.TripletPrecision(
            inputs=("targetPositiveLogits", "targetNegativeLogits", "targetMask", "targetWeight"),
            outputs="triplet_precision",
        ),
        deepr.layers.Select(inputs=("loss", "triplet_precision")),
    )
