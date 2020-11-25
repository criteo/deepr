"""Compute prediction on a dataset and log result."""

import logging
from typing import List, Callable, Union, Optional
from dataclasses import dataclass

import tensorflow as tf

import deepr


LOGGER = logging.getLogger(__name__)


@dataclass
class PredictProto(deepr.jobs.Job):
    """Compute predictions from a single .pb file."""

    path_model: str
    graph_name: str
    input_fn: Callable[[], tf.data.Dataset]
    prepro_fn: Callable[[tf.data.Dataset, str], tf.data.Dataset]
    feeds: Union[str, List[str]]
    fetches: Union[str, List[str]]

    def run(self):
        predictor = deepr.predictors.ProtoPredictor(
            path=f"{self.path_model}/{self.graph_name}", feeds=self.feeds, fetches=self.fetches
        )
        for preds in predictor(lambda: self.prepro_fn(self.input_fn(), tf.estimator.ModeKeys.PREDICT)):
            LOGGER.info(preds)


@dataclass
class PredictSavedModel(deepr.jobs.Job):
    """Compute predictions from a directory containing saved models."""

    path_saved_model: str
    input_fn: Callable[[], tf.data.Dataset]
    prepro_fn: Callable[[tf.data.Dataset, str], tf.data.Dataset]
    feeds: Optional[Union[str, List[str]]] = None
    fetches: Optional[Union[str, List[str]]] = None

    def run(self):
        predictor = deepr.predictors.SavedModelPredictor(
            path=deepr.predictors.get_latest_saved_model(self.path_saved_model), feeds=self.feeds, fetches=self.fetches
        )
        for preds in predictor(lambda: self.prepro_fn(self.input_fn(), tf.estimator.ModeKeys.PREDICT)):
            LOGGER.info(preds)
