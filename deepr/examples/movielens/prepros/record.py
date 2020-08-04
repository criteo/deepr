# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
"""TF Record Preprocessing for MovieLens."""

from typing import Optional

import deepr as dpr
import tensorflow as tf

from deepr.examples.movielens.utils import fields


FIELDS_RECORD = [fields.UID, fields.INPUT_POSITIVES, fields.TARGET_POSITIVES, fields.TARGET_NEGATIVES]

FIELDS_PREPRO = [fields.INPUT_MASK, fields.TARGET_MASK]


def RecordPrepro(
    min_input_size: int = 3,
    min_target_size: int = 3,
    max_input_size: int = 50,
    max_target_size: int = 50,
    buffer_size: int = 1024,
    batch_size: int = 128,
    repeat_size: Optional[int] = None,
    prefetch_size: int = 1,
    num_parallel_calls: int = 8,
):
    """Default Preprocessing for MovieLens."""
    return dpr.prepros.Serial(
        dpr.prepros.FromExample(FIELDS_RECORD),
        (
            dpr.prepros.Map(dpr.layers.ToDense(field.default, inputs=field.name, outputs=field.name))
            for field in FIELDS_RECORD
            if field.is_sparse()
        ),
        dpr.prepros.Filter(
            dpr.layers.IsMinSize(inputs="inputPositives", size=min_input_size), modes=[dpr.TRAIN, dpr.EVAL]
        ),
        dpr.prepros.Filter(
            dpr.layers.IsMinSize(inputs="targetPositives", size=min_target_size), modes=[dpr.TRAIN, dpr.EVAL]
        ),
        dpr.prepros.Map(dpr.layers.SliceLast(max_input_size, inputs="inputPositives", outputs="inputPositives")),
        dpr.prepros.Map(dpr.layers.SliceFirst(max_target_size, inputs="targetPositives", outputs="targetPositives")),
        dpr.prepros.Map(dpr.layers.SliceFirst(max_target_size, inputs="targetNegatives", outputs="targetNegatives")),
        dpr.prepros.Map(SequenceMask(inputs="inputPositives", outputs="inputMask")),
        dpr.prepros.Map(SequenceMask(inputs="targetPositives", outputs="targetMask")),
        dpr.prepros.Shuffle(buffer_size=buffer_size, modes=[dpr.TRAIN]),
        (dpr.prepros.PaddedBatch(batch_size=batch_size, fields=FIELDS_RECORD + FIELDS_PREPRO)),
        dpr.prepros.Repeat(repeat_size, modes=[dpr.TRAIN]),
        dpr.prepros.Prefetch(prefetch_size),
        num_parallel_calls=num_parallel_calls,
    )


@dpr.layers.layer(n_in=1, n_out=1)
def SequenceMask(tensors):
    size = tf.size(tensors)
    return tf.sequence_mask(size)
