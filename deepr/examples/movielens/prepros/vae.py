# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
"""TF Record Preprocessing for MovieLens."""

from typing import Optional

import deepr as dpr
import tensorflow as tf

from deepr.examples.movielens.utils import fields


FIELDS = [fields.UID, fields.INPUT_POSITIVES, fields.INPUT_MASK]


def VAEPrepro(
    buffer_size: int = 1024,
    batch_size: int = 512,
    repeat_size: Optional[int] = None,
    prefetch_size: int = 1,
    num_parallel_calls: int = 8,
):
    """Default Preprocessing for MovieLens."""
    return dpr.prepros.Serial(
        dpr.prepros.Map(SequenceMask(inputs="inputPositives", outputs="inputMask")),
        dpr.prepros.Shuffle(buffer_size=buffer_size, modes=[dpr.TRAIN]),
        (dpr.prepros.PaddedBatch(batch_size=batch_size, fields=FIELDS)),
        dpr.prepros.Repeat(repeat_size, modes=[dpr.TRAIN]),
        dpr.prepros.Prefetch(prefetch_size),
        num_parallel_calls=num_parallel_calls,
    )


@dpr.layers.layer(n_in=1, n_out=1)
def SequenceMask(tensors):
    size = tf.size(tensors)
    return tf.sequence_mask(size)
