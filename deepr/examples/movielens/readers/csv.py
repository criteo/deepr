# pylint: disable=invalid-name
"""CSV Reader for MovieLens."""

import logging
import collections

import tensorflow as tf
import numpy as np
import deepr

from deepr.examples.movielens.utils import fields


try:
    import pandas as pd
except ImportError as e:
    print(f"Pandas needs to be installed for MovieLens {e}")


try:
    from scipy import sparse
except ImportError as e:
    print(f"Scipy needs to be installed for MovieLens {e}")


LOGGER = logging.getLogger(__name__)


class TrainCSVReader(deepr.readers.Reader):
    """Reader of MovieLens CSV files of the Multi-VAE paper.

    See https://github.com/dawenl/vae_cf
    """

    def __init__(
        self,
        path_csv: str,
        vocab_size: int,
        target_ratio: float = None,
        bucket_size: int = 16 * 512,
        shuffle: bool = True,
        seed: int = 42,
        take_ratio: float = None,
    ):
        self.path_csv = path_csv
        self.vocab_size = vocab_size
        self.target_ratio = target_ratio
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.seed = seed
        self.take_ratio = take_ratio
        self.fields = [
            fields.UID,
            fields.INPUT_POSITIVES,
            fields.TARGET_POSITIVES,
            fields.INPUT_POSITIVES_ONE_HOT(vocab_size),
            fields.TARGET_POSITIVES_ONE_HOT(vocab_size),
        ]

    def as_dataset(self):
        with deepr.io.Path(self.path_csv).open() as file:
            tp = pd.read_csv(file)
        rows, cols = tp["uid"], tp["sid"]
        n_users = rows.max() + 1
        data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype="int64", shape=(n_users, self.vocab_size))
        LOGGER.info(f"Reloaded user-item matrix of shape {data.shape} with num_events={len(rows)}")
        if self.take_ratio is not None:
            data = data[: int(self.take_ratio * n_users)]
            LOGGER.info(f"Sliced user-item matrix, new shape = {data.shape}, num_events={data.sum()}")
        np.random.seed(self.seed)
        counts = np.array(data.sum(axis=1)).flatten()
        if self.bucket_size:
            buckets = collections.defaultdict(list)
            for bucket, (_, idx) in enumerate(sorted(zip(counts, range(data.shape[0])))):
                buckets[bucket // self.bucket_size].append(idx)

        def _gen():
            # Resolve idxlist (shuffle + buckets)
            if self.shuffle:
                if self.bucket_size:
                    idxlist = []
                    # Shuffle buckets
                    bucket_idx = list(range(len(buckets)))
                    np.random.shuffle(bucket_idx)
                    for idx in bucket_idx:
                        # Shuffle bucket indices
                        bucket_idxlist = buckets[idx]
                        np.random.shuffle(bucket_idxlist)
                        idxlist.extend(bucket_idxlist)
                else:
                    idxlist = list(range(data.shape[0]))
                    np.random.shuffle(idxlist)
            else:
                if self.bucket_size:
                    idxlist = [idx for _, idx in sorted(zip(counts, range(data.shape[0])))]
                else:
                    idxlist = list(range(data.shape[0]))

            # Iterate over items in the sparse matrix
            for idx in idxlist:
                X = data[idx]
                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype("int64")
                indices = np.nonzero(X[0])[0]
                # Split into input / target
                if self.target_ratio is not None:
                    np.random.shuffle(indices)
                    size = int(len(indices) * (1 - self.target_ratio))
                    input_positives = indices[:size]
                    target_positives = indices[size:]
                    input_one_hot = np.zeros((self.vocab_size,), dtype=np.int64)
                    input_one_hot[input_positives] = 1
                    target_one_hot = np.zeros((self.vocab_size,), dtype=np.int64)
                    target_one_hot[target_positives] = 1
                # Same input / target
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


class TestCSVReader(deepr.readers.Reader):
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
        with deepr.io.Path(self.path_csv_tr).open() as file:
            tp_tr = pd.read_csv(file)
        with deepr.io.Path(self.path_csv_te).open() as file:
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
