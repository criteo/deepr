"""CSV Reader for MovieLens."""

import pandas as pd

import tensorflow as tf
import deepr as dpr

from deepr.examples.movielens.utils import fields


FIELDS = [fields.UID, fields.INPUT_POSITIVES]


class VAEReader(dpr.readers.Reader):
    """CSV Reader for MovieLens."""

    def __init__(self, path_csv: str):
        self.path_csv = path_csv

    def as_dataset(self):
        data = pd.read_csv(self.path_csv)
        df = data.groupby("uid").agg(list).reset_index()

        def _gen():
            for _, row in df.iterrows():
                yield {"uid": row.uid, "inputPositives": row.sid}

        return tf.data.Dataset.from_generator(
            _gen,
            output_types={field.name: field.dtype for field in FIELDS},
            output_shapes={field.name: field.shape for field in FIELDS},
        )
