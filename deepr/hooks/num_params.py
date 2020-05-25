"""Log Number of Parameters after session creation"""

from typing import Tuple, List
import logging

import tensorflow as tf

from deepr.utils import mlflow


LOGGER = logging.getLogger(__name__)


class NumParamsHook(tf.train.SessionRunHook):
    """Log Number of Parameters after session creation"""

    def __init__(self, use_mlflow: bool = False):
        self.use_mlflow = use_mlflow

    def after_create_session(self, session, coord):
        super().after_create_session(session, coord)
        num_global, num_trainable = get_num_params()
        LOGGER.info(f"Number of parameters (global) = {num_global}")
        LOGGER.info(f"Number of parameters (trainable) = {num_trainable}")
        if self.use_mlflow:
            mlflow.log_metrics({"num_params_global": num_global, "num_params_trainable": num_trainable})


def get_num_params() -> Tuple[int, int]:
    """Get number of global and trainable parameters

    Returns
    -------
    Tuple[int, int]
        num_global, num_trainable
    """

    def _count(variables: List):
        total = 0
        for var in variables:
            shape = var.get_shape()
            var_params = 1
            for dim in shape:
                var_params *= dim.value
            total += var_params
        return total

    num_global = _count(tf.global_variables())
    num_trainable = _count(tf.trainable_variables())
    return num_global, num_trainable
