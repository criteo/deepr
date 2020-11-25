# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
"""CSV Preprocessing for MovieLens."""

from typing import Optional

import deepr
import tensorflow as tf

from deepr.examples.movielens.utils import fields as F


def CSVPrepro(
    vocab_size: int,
    batch_size: int = 512,
    repeat_size: Optional[int] = None,
    prefetch_size: int = 1,
    num_parallel_calls: int = 8,
    num_negatives: int = None,
):
    """CSV Preprocessing for MovieLens."""
    fields = [
        F.UID,
        F.INPUT_POSITIVES,
        F.INPUT_MASK,
        F.TARGET_POSITIVES,
        F.TARGET_MASK,
        F.INPUT_POSITIVES_ONE_HOT(vocab_size),
        F.TARGET_POSITIVES_ONE_HOT(vocab_size),
    ]
    return deepr.prepros.Serial(
        deepr.prepros.Map(SequenceMask(inputs="inputPositives", outputs="inputMask")),
        deepr.prepros.Map(SequenceMask(inputs="targetPositives", outputs="targetMask")),
        deepr.prepros.PaddedBatch(batch_size=batch_size, fields=fields),
        deepr.prepros.Map(
            RandomNegatives(
                inputs="targetPositives", outputs="targetNegatives", num_negatives=num_negatives, vocab_size=vocab_size
            )
        )
        if num_negatives is not None
        else [],
        deepr.prepros.Repeat(repeat_size, modes=[deepr.TRAIN]),
        deepr.prepros.Prefetch(prefetch_size),
        num_parallel_calls=num_parallel_calls,
    )


@deepr.layers.layer(n_in=1, n_out=1)
def SequenceMask(tensors):
    size = tf.size(tensors)
    return tf.sequence_mask(size)


@deepr.layers.layer(n_in=1, n_out=1)
def RandomNegatives(tensors, num_negatives, vocab_size):
    negatives = tf.random.uniform(shape=[tf.shape(tensors)[0], 1, num_negatives], maxval=vocab_size, dtype=tf.int64)
    return negatives
