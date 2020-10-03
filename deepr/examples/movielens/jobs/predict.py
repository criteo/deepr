"""Compute MovieLens predictions."""

import logging
from typing import Callable
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import pyarrow as pa

import deepr as dpr

try:
    import pandas as pd
except ImportError as e:
    print(f"Pandas needs to be installed for MovieLens {e}")


LOGGER = logging.getLogger(__name__)


COLUMNS = ["uid", "user", "input", "target"]


SCHEMA = pa.schema(
    [
        ("uid", pa.int64()),
        ("user", pa.list_(pa.float32())),
        ("input", pa.list_(pa.int64())),
        ("target", pa.list_(pa.int64())),
    ]
)


@dataclass
class Predict(dpr.jobs.Job):
    """Compute MovieLens predictions."""

    path_saved_model: str
    path_predictions: str
    input_fn: Callable[[], tf.data.Dataset]
    prepro_fn: Callable[[tf.data.Dataset, str], tf.data.Dataset]

    def run(self):
        LOGGER.info(f"Computing predictions from {self.path_saved_model}")
        predictor = dpr.predictors.SavedModelPredictor(
            path=dpr.predictors.get_latest_saved_model(self.path_saved_model)
        )
        predictions = []
        for preds in predictor(lambda: self.prepro_fn(self.input_fn(), tf.estimator.ModeKeys.PREDICT)):
            for uid, user, input_idx, input_mask, target_idx, target_mask in zip(
                preds["uid"],
                preds["userEmbeddings"],
                preds["inputPositives"],
                preds["inputMask"],
                preds["targetPositives"],
                preds["targetMask"],
            ):
                predictions.append(
                    (
                        uid,
                        user.astype(np.float32).tolist(),
                        input_idx[input_mask].astype(np.int64).tolist(),
                        target_idx[target_mask].astype(np.int64).tolist(),
                    )
                )

        with dpr.io.ParquetDataset(self.path_predictions).open() as ds:
            df = pd.DataFrame(data=predictions, columns=COLUMNS)
            ds.write_pandas(df, compression="snappy", schema=SCHEMA)
        LOGGER.info(f"Wrote predictions to {self.path_predictions}")


@dataclass
class PredictSVD(dpr.jobs.Job):
    """Compute MovieLens predictions."""

    path_embeddings: str
    path_predictions: str
    input_fn: Callable[[], tf.data.Dataset]
    prepro_fn: Callable[[tf.data.Dataset, str], tf.data.Dataset]
    normalize: bool = False

    def run(self):
        LOGGER.info(f"Reloading embeddings from {self.path_embeddings}")
        with dpr.io.Path(self.path_embeddings).open("rb") as file:
            embeddings = np.load(file)
            embeddings = embeddings.astype(np.float32)

        if self.normalize:
            LOGGER.info("Normalizing product embeddings.")
            embeddings = np.divide(embeddings, np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        predictions = []
        for batch in dpr.readers.from_dataset(self.prepro_fn(self.input_fn(), tf.estimator.ModeKeys.PREDICT)):
            for uid, input_idx, input_mask, target_idx, target_mask in zip(
                batch["uid"],
                batch["inputPositives"],
                batch["inputMask"],
                batch["targetPositives"],
                batch["targetMask"],
            ):
                input_indices = input_idx[input_mask].astype(np.int64)
                target_indices = target_idx[target_mask].astype(np.int64)
                user = np.sum(embeddings[input_indices], axis=0).astype(np.float32)
                predictions.append(
                    (
                        uid,
                        user.tolist(),
                        input_indices.tolist(),
                        target_indices.tolist(),
                    )
                )

        with dpr.io.ParquetDataset(self.path_predictions).open() as ds:
            df = pd.DataFrame(data=predictions, columns=COLUMNS)
            ds.write_pandas(df, compression="snappy", schema=SCHEMA)
        LOGGER.info(f"Wrote predictions to {self.path_predictions}")
