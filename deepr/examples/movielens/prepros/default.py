# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
"""Description"""

from typing import Optional

import deepr as dpr
import tensorflow as tf

from deepr.examples.movielens.utils import fields


FIELDS_RECORD = [fields.UID, fields.INPUT_POSITIVES, fields.TARGET_POSITIVES, fields.TARGET_NEGATIVES]

FIELDS_PREPRO = [fields.INPUT_MASK, fields.TARGET_MASK]


def DefaultPrepro(
    batch_size: int = 16,
    buffer_size: int = 10,
    epochs: Optional[int] = None,
    max_input_size: int = 10000,
    max_target_size: int = 1000,
):
    sparse_fields = [field for field in FIELDS_RECORD if field.is_sparse()]
    return dpr.prepros.Serial(
        dpr.prepros.FromExample(FIELDS_RECORD),
        (dpr.prepros.Map(dpr.layers.ToDense(f.default, inputs=f.name, outputs=f.name)) for f in sparse_fields),
        (
            dpr.prepros.Map(dpr.layers.SliceLast(max_input_size, inputs="inputPositives", outputs=key))
            for key in ["inputPositives"]
        ),
        (
            dpr.prepros.Map(dpr.layers.SliceFirst(max_target_size, inputs=key, outputs=key))
            for key in ["targetPositives", "targetNegatives"]
        ),
        dpr.prepros.Map(SequenceMask(inputs="inputPositives", outputs="inputMask")),
        dpr.prepros.Map(SequenceMask(inputs="targetPositives", outputs="targetMask")),
        (dpr.prepros.PaddedBatch(batch_size=batch_size, fields=FIELDS_RECORD + FIELDS_PREPRO)),
        dpr.prepros.Repeat(epochs, modes=[dpr.TRAIN]),
        dpr.prepros.Prefetch(buffer_size),
    )


@dpr.layers.layer(n_in=1, n_out=1)
def SequenceMask(tensors):
    size = tf.size(tensors)
    return tf.sequence_mask(size)
