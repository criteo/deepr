"""Build MovieLens dataset as TFRecords."""

import logging
from dataclasses import dataclass

import numpy as np

import deepr as dpr

try:
    import pandas as pd
except ImportError as e:
    print(f"Pandas needs to be installed for MovieLens {e}")

try:
    from scipy import sparse
except ImportError as e:
    print(f"Scipy needs to be installed for MovieLens {e}")


try:
    from sklearn.decomposition import TruncatedSVD
except ImportError as e:
    print(f"sklearn needs to be installed for MovieLens {e}")


LOGGER = logging.getLogger(__name__)


@dataclass
class SVD(dpr.jobs.Job):
    """Build SVD embeddings."""

    path_csv: str
    path_embeddings: str
    vocab_size: int
    dim: int = 100
    min_count: int = 10

    def run(self):
        # Read user-item matrix
        LOGGER.info(f"Reading user-item rankings from {self.path_csv}")
        with dpr.io.Path(self.path_csv).open() as file:
            tp = pd.read_csv(file)
        n_users = tp["uid"].max() + 1
        rows, cols = tp["uid"], tp["sid"]
        user_item = sparse.csr_matrix(
            (np.ones_like(rows), (rows, cols)), dtype="int64", shape=(n_users, self.vocab_size)
        )

        # Computing co-occurrence matrix
        LOGGER.info("Computing co-occurrence matrix")
        item_item = compute_coocurrence(user_item, self.min_count)

        # Compute PMI from co-occurrence
        LOGGER.info("Computing PMI matrix")
        pmi = compute_pmi(item_item)

        # Compute Truncated SVD
        LOGGER.info("Computing Truncated SVD from PMI matrix")
        svd = TruncatedSVD(n_components=self.dim, algorithm="arpack", random_state=42)
        svd.fit(pmi)
        LOGGER.info(f"Explained variance: {svd.explained_variance_ratio_.sum()}")

        LOGGER.info(f"Saving embeddings to {self.path_embeddings}")
        dpr.io.Path(self.path_embeddings).parent.mkdir(parents=True, exist_ok=True)
        embeddings = svd.transform(pmi)
        with dpr.io.Path(self.path_embeddings).open("wb") as file:
            np.save(file, embeddings)


def compute_coocurrence(user_item, min_count: int):
    """Compute co-occurrence matrix from user-item matrix."""
    # Compute co-occurrences via dot-product (all entries are 1 or 0)
    item_item = user_item.transpose().dot(user_item)

    # Set diagonal to zero
    item_item.setdiag(0)

    # Get indices to mask
    counts = item_item.sum(axis=-1).A1
    positive = counts > 0
    rare = counts < min_count
    mask = np.logical_and(positive, rare)
    indices = np.nonzero(mask)

    # Set rows / columns to zeros
    item_item = item_item.tolil()
    item_item[indices] = 0
    item_item = item_item.transpose()
    item_item[indices] = 0

    # Convert back to CSR format and remove zeros
    item_item = item_item.tocsr()
    item_item.eliminate_zeros()

    return item_item


def compute_pmi(matrix, cds: float = 0.75, additive_smoothing: float = 0.0, pmi_power: float = 1.0, k=1.0):
    """Compute PMI matrix from item-item matrix."""
    # Convert to COO format
    matrix_tocoo = matrix.tocoo()
    data = matrix_tocoo.data
    rows = matrix_tocoo.row
    cols = matrix_tocoo.col

    # Compute items counts
    left = np.array(matrix.sum(axis=1)).flatten()
    right = np.array(matrix.sum(axis=0)).flatten()

    # Compute total counts
    total_count = data.sum()
    smoothed_total_count = np.power(data, cds).sum()
    scaled_smoothing = np.power(additive_smoothing, cds)

    # Compute probabilities
    p_xy = np.power(data, pmi_power) / total_count
    p_x = (left[rows] + additive_smoothing) / (total_count + additive_smoothing)
    p_y_cds = np.power(right[cols] + additive_smoothing, cds) / (smoothed_total_count + scaled_smoothing)

    # Compute PMI and assemble into on CSR matrix
    data_pmi = np.maximum(np.log(p_xy) - np.log(p_x) - np.log(p_y_cds) - np.log(k), 0)
    pmi = sparse.csr_matrix((data_pmi, (rows, cols)), shape=matrix.shape)
    return pmi
