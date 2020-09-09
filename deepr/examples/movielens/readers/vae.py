"""CSV Reader for MovieLens."""

import pandas as pd
from scipy import sparse
import tensorflow as tf
import numpy as np
import deepr as dpr

from deepr.examples.movielens.utils import fields


FIELDS = [fields.UID, fields.INPUT_POSITIVES]


class VAEReader(dpr.readers.Reader):
    """CSV Reader for MovieLens."""

    def __init__(self, path_csv: str, vocab_size: int):
        self.path_csv = path_csv
        self.vocab_size = vocab_size
        self.fields = FIELDS + [dpr.Field(name="one_hot", shape=(vocab_size,), dtype=tf.int32)]

    def as_dataset(self):
        tp = pd.read_csv(self.path_csv)
        n_users = tp["uid"].max() + 1
        rows, cols = tp["uid"], tp["sid"]
        data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype="float64", shape=(n_users, self.vocab_size))

        def _gen():
            for idx in range(data.shape[0]):
                X = data[idx]
                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype("float32")
                yield {"uid": idx, "inputPositives": np.nonzero(X[0])[0], "one_hot": X[0]}

        return tf.data.Dataset.from_generator(
            _gen,
            output_types={field.name: field.dtype for field in self.fields},
            output_shapes={field.name: field.shape for field in self.fields},
        )


class TestVAEReader(dpr.readers.Reader):
    """Test VAE Reader."""

    def __init__(self, path_csv_tr: str, path_csv_te: str, vocab_size: int):
        self.path_csv_tr = path_csv_tr
        self.path_csv_te = path_csv_te
        self.vocab_size = vocab_size
        self.fields = FIELDS + [fields.TARGET_POSITIVES, dpr.Field(name="one_hot", shape=(vocab_size,), dtype=tf.int32)]

    def as_dataset(self):
        tp_tr = pd.read_csv(self.path_csv_tr)
        tp_te = pd.read_csv(self.path_csv_te)

        start_idx = min(tp_tr["uid"].min(), tp_te["uid"].min())
        end_idx = max(tp_tr["uid"].max(), tp_te["uid"].max())

        rows_tr, cols_tr = tp_tr["uid"] - start_idx, tp_tr["sid"]
        rows_te, cols_te = tp_te["uid"] - start_idx, tp_te["sid"]

        data_tr = sparse.csr_matrix(
            (np.ones_like(rows_tr), (rows_tr, cols_tr)),
            dtype="float64",
            shape=(end_idx - start_idx + 1, self.vocab_size),
        )
        data_te = sparse.csr_matrix(
            (np.ones_like(rows_te), (rows_te, cols_te)),
            dtype="float64",
            shape=(end_idx - start_idx + 1, self.vocab_size),
        )

        def _gen():
            for idx in range(data_tr.shape[0]):
                X = data_tr[idx]
                y = data_te[idx]
                if sparse.isspmatrix(X):
                    X = X.toarray()
                if sparse.isspmatrix(y):
                    y = y.toarray()
                X = X.astype("float32")
                y = y.astype("float32")
                yield {
                    "uid": idx,
                    "inputPositives": np.nonzero(X[0])[0],
                    "one_hot": X[0],
                    "targetPositives": np.nonzero(y[0])[0],
                }

        return tf.data.Dataset.from_generator(
            _gen,
            output_types={field.name: field.dtype for field in self.fields},
            output_shapes={field.name: field.shape for field in self.fields},
        )
