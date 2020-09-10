# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
"""CSV Preprocessing for MovieLens."""

from typing import Optional

import deepr as dpr
import tensorflow as tf

from deepr.examples.movielens.utils import fields as F


def CSVPrepro(
    vocab_size: int,
    test: bool = False,
    buffer_size: int = 1024,
    batch_size: int = 512,
    repeat_size: Optional[int] = None,
    prefetch_size: int = 1,
    num_parallel_calls: int = 8,
):
    """CSV Preprocessing for MovieLens."""
    fields = [
        F.UID,
        F.INPUT_POSITIVES,
        F.INPUT_MASK,
        F.INPUT_POSITIVES_ONE_HOT(vocab_size),
    ]
    if test:
        fields += [F.TARGET_POSITIVES, F.TARGET_MASK]
    return dpr.prepros.Serial(
        dpr.prepros.Map(SequenceMask(inputs="inputPositives", outputs="inputMask")),
        dpr.prepros.Map(SequenceMask(inputs="targetPositives", outputs="targetMask")) if test else [],
        dpr.prepros.Shuffle(buffer_size=buffer_size, modes=[dpr.TRAIN]),
        dpr.prepros.PaddedBatch(batch_size=batch_size, fields=fields),
        dpr.prepros.Repeat(repeat_size, modes=[dpr.TRAIN]),
        dpr.prepros.Prefetch(prefetch_size),
        num_parallel_calls=num_parallel_calls,
    )


@dpr.layers.layer(n_in=1, n_out=1)
def SequenceMask(tensors):
    size = tf.size(tensors)
    return tf.sequence_mask(size)
