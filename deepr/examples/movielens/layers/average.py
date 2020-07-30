# pylint: disable=unexpected-keyword-arg,no-value-for-parameter
"""Average Model."""

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
    )
