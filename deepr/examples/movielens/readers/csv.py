# pylint: disable=invalid-name
"""CSV Reader for MovieLens."""

import pandas as pd
from scipy import sparse
import tensorflow as tf
import numpy as np
import deepr as dpr

from deepr.examples.movielens.utils import fields


class CSVReader(dpr.readers.Reader):
    """Reader of MovieLens CSV files of the Multi-VAE paper.

    See https://github.com/dawenl/vae_cf
    """

    def __init__(self, path_csv: str, vocab_size: int):
        self.path_csv = path_csv
        self.vocab_size = vocab_size
        self.fields = [fields.UID, fields.INPUT_POSITIVES, fields.INPUT_POSITIVES_ONE_HOT(vocab_size)]

    def as_dataset(self):
        tp = pd.read_csv(self.path_csv)
        n_users = tp["uid"].max() + 1
        rows, cols = tp["uid"], tp["sid"]
        data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype="int64", shape=(n_users, self.vocab_size))

        def _gen():
            for idx in range(data.shape[0]):
                X = data[idx]
                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype("int64")
                yield {"uid": idx, "inputPositives": np.nonzero(X[0])[0], "inputPositivesOneHot": X[0]}

        return tf.data.Dataset.from_generator(
            _gen,
            output_types={field.name: field.dtype for field in self.fields},
            output_shapes={field.name: field.shape for field in self.fields},
        )


class TestCSVReader(dpr.readers.Reader):
    """Reader of MovieLens test CSV files of the Multi-VAE paper.

    See https://github.com/dawenl/vae_cf
    """

    def __init__(self, path_csv_tr: str, path_csv_te: str, vocab_size: int):
        self.path_csv_tr = path_csv_tr
        self.path_csv_te = path_csv_te
        self.vocab_size = vocab_size
        self.fields = [
            fields.UID,
            fields.INPUT_POSITIVES,
            fields.INPUT_POSITIVES_ONE_HOT(vocab_size),
            fields.TARGET_POSITIVES,
        ]

    def as_dataset(self):
        tp_tr = pd.read_csv(self.path_csv_tr)
        tp_te = pd.read_csv(self.path_csv_te)

        start_idx = min(tp_tr["uid"].min(), tp_te["uid"].min())
        end_idx = max(tp_tr["uid"].max(), tp_te["uid"].max())

        rows_tr, cols_tr = tp_tr["uid"] - start_idx, tp_tr["sid"]
        rows_te, cols_te = tp_te["uid"] - start_idx, tp_te["sid"]

        data_tr = sparse.csr_matrix(
            (np.ones_like(rows_tr), (rows_tr, cols_tr)),
            dtype="int64",
            shape=(end_idx - start_idx + 1, self.vocab_size),
        )
        data_te = sparse.csr_matrix(
            (np.ones_like(rows_te), (rows_te, cols_te)),
            dtype="int64",
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
                X = X.astype("int64")
                y = y.astype("int64")
                yield {
                    "uid": idx,
                    "inputPositives": np.nonzero(X[0])[0],
                    "inputPositivesOneHot": X[0],
                    "targetPositives": np.nonzero(y[0])[0],
                }

        return tf.data.Dataset.from_generator(
            _gen,
            output_types={field.name: field.dtype for field in self.fields},
            output_shapes={field.name: field.shape for field in self.fields},
        )
