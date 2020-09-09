# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
"""TF Record Preprocessing for MovieLens."""

from typing import Optional

import deepr as dpr
import tensorflow as tf

from deepr.examples.movielens.utils import fields


FIELDS = [fields.UID, fields.INPUT_POSITIVES, fields.INPUT_MASK]


def VAEPrepro(
    vocab_size: int,
    buffer_size: int = 1024,
    batch_size: int = 512,
    repeat_size: Optional[int] = None,
    prefetch_size: int = 1,
    num_parallel_calls: int = 8,
    test: bool = False,
):
    """Default Preprocessing for MovieLens."""
    if not test:
        _fields = FIELDS + [dpr.Field(name="one_hot", shape=(vocab_size,), dtype=tf.int32)]
    else:
        _fields = FIELDS + [
            fields.TARGET_POSITIVES,
            fields.TARGET_MASK,
            dpr.Field(name="one_hot", shape=(vocab_size,), dtype=tf.int32),
        ]
    return dpr.prepros.Serial(
        dpr.prepros.Map(SequenceMask(inputs="inputPositives", outputs="inputMask")),
        dpr.prepros.Map(SequenceMask(inputs="targetPositives", outputs="targetMask")) if test else [],
        dpr.prepros.Shuffle(buffer_size=buffer_size, modes=[dpr.TRAIN]),
        (dpr.prepros.PaddedBatch(batch_size=batch_size, fields=_fields)),
        dpr.prepros.Repeat(repeat_size, modes=[dpr.TRAIN]),
        dpr.prepros.Prefetch(prefetch_size),
        num_parallel_calls=num_parallel_calls,
    )


@dpr.layers.layer(n_in=1, n_out=1)
def SequenceMask(tensors):
    size = tf.size(tensors)
    return tf.sequence_mask(size)
