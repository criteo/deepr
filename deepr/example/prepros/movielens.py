"""Description"""

from typing import Optional

import deepr as dpr

from deepr.example.utils import fields


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
        # TODO: add mask
        (dpr.prepros.PaddedBatch(batch_size=batch_size, fields=fields)),
        dpr.prepros.Repeat(epochs, modes=[dpr.TRAIN]),
        dpr.prepros.Prefetch(buffer_size),
    )
