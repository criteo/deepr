# pylint: disable=no-value-for-parameter
"""Eval dataset."""

import logging
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np

import deepr as dpr


LOGGER = logging.getLogger(__name__)


@dataclass
class Evaluate(dpr.jobs.Job):
    """Evaluate MovieLens."""

    path_predictions: str
    path_embeddings: str
    k: int

    def run(self):
        with dpr.io.ParquetDataset(self.path_predictions).open() as ds:
            predictions = ds.read_pandas().to_pandas()
            users = np.stack(predictions["user"])

        with dpr.io.ParquetDataset(self.path_embeddings).open() as ds:
            embeddings = ds.read_pandas().to_pandas()
            embeddings = embeddings.to_numpy()

        index = faiss.IndexFlatIP(embeddings.shape[-1])
        index.add(np.ascontiguousarray(embeddings))
        _, indices = index.search(users, k=self.k)
        precision, recall, f1_score = self.compute_metrics(predictions["target"], indices)
        LOGGER.info(
            f"precision@{self.k} = {precision}\n" f"recall@{self.k} = {recall}\n" f"f1_score@{self.k} = {f1_score}"
        )
        return precision, recall, f1_score

    def compute_metrics(self, actuals: List[np.ndarray], predictions: List[np.ndarray]):
        recalls = []
        precisions = []
        f1_scores = []
        for actual, pred in zip(actuals, predictions):
            p, r, f1 = self.precision_recall_f1(actual, pred)
            recalls.append(r)
            precisions.append(p)
            f1_scores.append(f1)
        return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

    @staticmethod
    def precision_recall_f1(true: np.ndarray, pred: np.ndarray):
        """ Compute precision, recall and f1_score"""
        num_predicted = np.unique(pred).size
        num_intersect = np.intersect1d(pred, true).size
        num_observed = np.unique(true).size
        p = num_intersect / num_predicted
        r = num_intersect / num_observed
        f1 = 2 * p * r / (p + r) if p != 0 or r != 0 else 0
        return p, r, f1