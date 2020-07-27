# pylint: disable=unexpected-keyword-arg,no-value-for-parameter
"""Transformer Model."""

import deepr as dpr


def TransformerModel(vocab_size: int, dim: int, **kwargs):
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("inputPositives", "inputMask")),
        dpr.layers.Embedding(
            inputs="inputPositives", outputs="inputEmbeddings", variable_name="embeddings", shape=[vocab_size, dim]
        ),
        dpr.layers.Transformer(
            inputs=("inputEmbeddings", "inputMask"),
            outputs="userEmbeddings",
            dim=dim,
            **kwargs
            ),
    )
