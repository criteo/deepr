# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
"""Description"""

from typing import Optional

import deepr as dpr
import tensorflow as tf

from deepr.example.utils import fields


INPUT_MASK = dpr.Field(name="inputMask", dtype=tf.bool, shape=[None], default=False)

TARGET_MASK = dpr.Field(name="targetMask", dtype=tf.bool, shape=[None], default=False)

ALL_FIELDS = fields.FIELDS_MOVIELENS + [INPUT_MASK, TARGET_MASK]


def MovieLensPrepro(
    batch_size: int = 16,
    buffer_size: int = 10,
    epochs: Optional[int] = None,
    max_input_size: int = 10000,
    max_target_size: int = 1000,
):
    sparse_fields = [field for field in fields.FIELDS_MOVIELENS if field.is_sparse()]
    return dpr.prepros.Serial(
        dpr.prepros.FromExample(fields.FIELDS_MOVIELENS),
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
        (dpr.prepros.PaddedBatch(batch_size=batch_size, fields=ALL_FIELDS)),
        dpr.prepros.Repeat(epochs, modes=[dpr.TRAIN]),
        dpr.prepros.Prefetch(buffer_size),
    )


@dpr.layers.layer(n_in=1, n_out=1)
def SequenceMask(tensors):
    size = tf.size(tensors)
    return tf.sequence_mask(size)
