# pylint: disable=no-value-for-parameter
"""Evaluate MovieLens."""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

import deepr as dpr
from deepr.utils import mlflow

try:
    import faiss
except ImportError as e:
    print(f"Faiss needs to be installed for MovieLens {e}")


LOGGER = logging.getLogger(__name__)


@dataclass
class Evaluate(dpr.jobs.Job):
    """Evaluate MovieLens."""

    path_predictions: str
    path_embeddings: str
    path_biases: str
    k: int
    use_mlflow: bool = False

    def run(self):
        with dpr.io.ParquetDataset(self.path_predictions).open() as ds:
            predictions = ds.read_pandas().to_pandas()
            users = np.stack(predictions["user"])
            ones = np.ones([users.shape[0], 1], np.float32)
            users_with_ones = np.concatenate([users, ones], axis=-1)

        with dpr.io.ParquetDataset(self.path_embeddings).open() as ds:
            embeddings = ds.read_pandas().to_pandas()
            embeddings = embeddings.to_numpy()

        with dpr.io.ParquetDataset(self.path_biases).open() as ds:
            biases = ds.read_pandas().to_pandas()
            biases = biases.to_numpy()

        embeddings_with_biases = np.concatenate([embeddings, biases], axis=-1)

        index = faiss.IndexFlatIP(embeddings_with_biases.shape[-1])
        index.add(np.ascontiguousarray(embeddings_with_biases))
        _, indices = index.search(users_with_ones, k=self.k)
        precision, recall, f1 = self.compute_metrics(predictions["target"], indices)
        LOGGER.info(f"precision@{self.k} = {precision}\n" f"recall@{self.k} = {recall}\n" f"f1@{self.k} = {f1}")
        if self.use_mlflow:
            mlflow.log_metric(key=f"precision_at_{self.k}", value=precision)
            mlflow.log_metric(key=f"recall_at_{self.k}", value=recall)
            mlflow.log_metric(key=f"f1_at_{self.k}", value=f1)

    def compute_metrics(self, actuals: List[np.ndarray], predictions: List[np.ndarray]):
        recalls = []
        precisions = []
        f1s = []
        for actual, pred in zip(actuals, predictions):
            p, r, f1 = self.precision_recall_f1(actual, pred)
            recalls.append(r)
            precisions.append(p)
            f1s.append(f1)
        return np.mean(precisions), np.mean(recalls), np.mean(f1s)

    @staticmethod
    def precision_recall_f1(true: np.ndarray, pred: np.ndarray):
        """Compute precision, recall and f1_score."""
        num_predicted = np.unique(pred).size
        num_intersect = np.intersect1d(pred, true).size
        num_observed = np.unique(true).size
        p = num_intersect / num_predicted
        r = num_intersect / num_observed
        f1 = 2 * p * r / (p + r) if p != 0 or r != 0 else 0
        return p, r, f1
