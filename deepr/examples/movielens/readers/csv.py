# pylint: disable=invalid-name
"""CSV Reader for MovieLens."""

import tensorflow as tf
import numpy as np
import deepr as dpr

from deepr.examples.movielens.utils import fields


try:
    import pandas as pd
except ImportError as e:
    print(f"Pandas needs to be installed for MovieLens {e}")


try:
    from scipy import sparse
except ImportError as e:
    print(f"Scipy needs to be installed for MovieLens {e}")


class TrainCSVReader(dpr.readers.Reader):
    """Reader of MovieLens CSV files of the Multi-VAE paper.

    See https://github.com/dawenl/vae_cf
    """

    def __init__(
        self, path_csv: str, vocab_size: int, target_ratio: float = None, shuffle: bool = True, seed: int = 98765
    ):
        self.path_csv = path_csv
        self.vocab_size = vocab_size
        self.target_ratio = target_ratio
        self.shuffle = shuffle
        self.seed = seed
        self.fields = [
            fields.UID,
            fields.INPUT_POSITIVES,
            fields.TARGET_POSITIVES,
            fields.INPUT_POSITIVES_ONE_HOT(vocab_size),
            fields.TARGET_POSITIVES_ONE_HOT(vocab_size),
        ]

    def as_dataset(self):
        with dpr.io.Path(self.path_csv).open() as file:
            tp = pd.read_csv(file)
        n_users = tp["uid"].max() + 1
        rows, cols = tp["uid"], tp["sid"]
        data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype="int64", shape=(n_users, self.vocab_size))
        np.random.seed(self.seed)

        def _gen():
            idxlist = list(range(data.shape[0]))
            if self.shuffle:
                np.random.shuffle(idxlist)
            for idx in idxlist:
                X = data[idx]
                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype("int64")
                indices = np.nonzero(X[0])[0]
                if self.target_ratio is not None:
                    np.random.shuffle(indices)
                    size = int(len(indices) * (1 - self.target_ratio))
                    input_positives = indices[:size]
                    target_positives = indices[size:]
                    input_one_hot = np.zeros((self.vocab_size,), dtype=np.int64)
                    input_one_hot[input_positives] = 1
                    target_one_hot = np.zeros((self.vocab_size,), dtype=np.int64)
                    target_one_hot[target_positives] = 1
                else:
                    input_positives = indices
                    target_positives = indices
                    input_one_hot = X[0]
                    target_one_hot = X[0]
                yield {
                    "uid": idx,
                    "inputPositives": input_positives,
                    "inputPositivesOneHot": input_one_hot,
                    "targetPositives": target_positives,
                    "targetPositivesOneHot": target_one_hot,
                }

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
            fields.TARGET_POSITIVES,
            fields.INPUT_POSITIVES_ONE_HOT(vocab_size),
            fields.TARGET_POSITIVES_ONE_HOT(vocab_size),
        ]

    def as_dataset(self):
        with dpr.io.Path(self.path_csv_tr).open() as file:
            tp_tr = pd.read_csv(file)
        with dpr.io.Path(self.path_csv_te).open() as file:
            tp_te = pd.read_csv(file)

        start_idx = min(tp_tr["uid"].min(), tp_te["uid"].min())
        end_idx = max(tp_tr["uid"].max(), tp_te["uid"].max())

        rows_tr, cols_tr = tp_tr["uid"] - start_idx, tp_tr["sid"]
        rows_te, cols_te = tp_te["uid"] - start_idx, tp_te["sid"]

        data_tr = sparse.csr_matrix(
            (np.ones_like(rows_tr), (rows_tr, cols_tr)), dtype="int64", shape=(end_idx - start_idx + 1, self.vocab_size)
        )
        data_te = sparse.csr_matrix(
            (np.ones_like(rows_te), (rows_te, cols_te)), dtype="int64", shape=(end_idx - start_idx + 1, self.vocab_size)
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
                    "targetPositivesOneHot": y[0],
                }

        return tf.data.Dataset.from_generator(
            _gen,
            output_types={field.name: field.dtype for field in self.fields},
            output_shapes={field.name: field.shape for field in self.fields},
        )
