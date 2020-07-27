"""Compute prediction on a dataset and log result."""

import logging
from typing import Callable
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import pyarrow as pa
import pandas as pd

import deepr as dpr


LOGGER = logging.getLogger(__name__)


COLUMNS = ["uid", "user", "target"]


SCHEMA = pa.schema([("uid", pa.string()), ("user", pa.list_(pa.float32())), ("target", pa.list_(pa.int64()))])


@dataclass
class Predict(dpr.jobs.Job):
    """Compute predictions from a directory containing saved models."""

    path_saved_model: str
    path_predictions: str
    input_fn: Callable[[], tf.data.Dataset]
    prepro_fn: Callable[[tf.data.Dataset, str], tf.data.Dataset]

    def run(self):
        predictor = dpr.predictors.SavedModelPredictor(
            path=dpr.predictors.get_latest_saved_model(self.path_saved_model)
        )
        predictions = []
        for preds in predictor(lambda: self.prepro_fn(self.input_fn(), tf.estimator.ModeKeys.PREDICT)):
            for uid, user, target, mask in zip(
                preds["uid"], preds["userEmbeddings"], preds["targetPositives"], preds["targetMask"]
            ):
                predictions.append((uid, user.astype(np.float32).tolist(), target[mask].astype(np.int64).tolist()))

        with dpr.io.ParquetDataset(self.path_predictions).open() as ds:
            df = pd.DataFrame(data=predictions, columns=COLUMNS)
            ds.write_pandas(df, compression="snappy", schema=SCHEMA)
