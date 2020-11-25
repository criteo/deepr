# pylint: disable=no-value-for-parameter
"""Evaluate MovieLens."""

import logging
from dataclasses import dataclass
from typing import List, Union, Optional

import numpy as np

import deepr
from deepr.utils import mlflow

try:
    import faiss
except ImportError as e:
    print(f"Faiss needs to be installed for MovieLens {e}")


LOGGER = logging.getLogger(__name__)


@dataclass
class Evaluate(deepr.jobs.Job):
    """Evaluate MovieLens using a Faiss Index.

    For each user embedding, the top num_queries items are retrieved.
    The input items are removed from the results, then we compare the
    remaining top-K results to the target items.
    """

    path_predictions: str
    path_embeddings: str
    path_biases: Optional[str] = None
    k: Union[int, List[int]] = 50
    use_mlflow: bool = False
    num_queries: int = 1000

    def run(self):
        with deepr.io.ParquetDataset(self.path_predictions).open() as ds:
            predictions = ds.read_pandas().to_pandas()
            users = np.stack(predictions["user"])

        if deepr.io.Path(self.path_embeddings).suffix == ".npz":
            with deepr.io.Path(self.path_embeddings).open("rb") as file:
                embeddings = np.load(file)
                embeddings = embeddings.astype(np.float32)
        else:
            with deepr.io.ParquetDataset(self.path_embeddings).open() as ds:
                embeddings = ds.read_pandas().to_pandas()
                embeddings = embeddings.to_numpy()

        if self.path_biases is not None:
            # Concatenate biases to product embeddings
            with deepr.io.ParquetDataset(self.path_biases).open() as ds:
                biases = ds.read_pandas().to_pandas()
                biases = biases.to_numpy()
            embeddings = np.concatenate([embeddings, biases], axis=-1)

            # Concatenate ones to users
            ones = np.ones([users.shape[0], 1], np.float32)
            users = np.concatenate([users, ones], axis=-1)

        LOGGER.info(f"Shapes, embeddings={embeddings.shape}, users={users.shape}")
        index = faiss.IndexFlatIP(embeddings.shape[-1])
        index.add(np.ascontiguousarray(embeddings))
        _, indices = index.search(users, k=self.num_queries)

        k_values = [self.k] if isinstance(self.k, int) else self.k
        for k in k_values:
            precision, recall, f1, ndcg = compute_metrics(predictions["input"], predictions["target"], indices, k=k)
            LOGGER.info(
                f"precision@{k} = {precision}\n" f"recall@{k} = {recall}\n" f"f1@{k} = {f1}\n" f"NDCG@{k} = {ndcg}"
            )
            if self.use_mlflow:
                mlflow.log_metric(key=f"precision_at_{k}", value=precision)
                mlflow.log_metric(key=f"recall_at_{k}", value=recall)
                mlflow.log_metric(key=f"f1_at_{k}", value=f1)
                mlflow.log_metric(key=f"ndcg_at_{k}", value=ndcg)


def compute_metrics(inputs: List[np.ndarray], targets: List[np.ndarray], predictions: List[np.ndarray], k: int):
    """Compute Recall, Precision and F1."""
    recalls = []
    precisions = []
    f1s = []
    ndcgs = []
    for inp, tgt, pred in zip(inputs, targets, predictions):
        # Remove indices that are in the input and take top k
        pred = [idx for idx in pred if idx not in inp][:k]
        p, r, f1 = precision_recall_f1(tgt, pred, k=k)
        ndcg = ndcg_score(tgt, pred, k=k)
        recalls.append(r)
        precisions.append(p)
        f1s.append(f1)
        ndcgs.append(ndcg)
    return np.mean(precisions), np.mean(recalls), np.mean(f1s), np.mean(ndcgs)


def precision_recall_f1(true: np.ndarray, pred: np.ndarray, k: int):
    """Compute precision, recall and f1_score."""
    num_predicted = np.unique(pred).size
    num_intersect = np.intersect1d(pred, true).size
    num_observed = np.unique(true).size
    p = num_intersect / min(num_predicted, k)
    r = num_intersect / min(num_observed, k)
    f1 = 2 * p * r / (p + r) if p != 0 or r != 0 else 0
    return p, r, f1


def ndcg_score(true: np.ndarray, pred: np.ndarray, k: int):
    """Compute Normalized Discounted Cumulative Gain."""
    tp = 1.0 / np.log2(np.arange(2, k + 2))
    correct = [1 if p in true else 0 for p in pred]
    dcg = np.sum(correct * tp)
    idcg = np.sum(tp[: min(len(true), k)])
    return dcg / idcg
