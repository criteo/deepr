"""Log Variables Statistics after initialization."""

import logging
from collections import defaultdict
import re
from typing import Tuple

import tensorflow as tf
import numpy as np

from deepr.utils import mlflow


LOGGER = logging.getLogger(__name__)


class LogVariablesInitHook(tf.train.SessionRunHook):
    """Log Variables Statistics after initialization."""

    def __init__(
        self,
        use_mlflow: bool = False,
        whitelist: Tuple[str, ...] = None,
        blacklist: Tuple[str, ...] = ("adam", "beta", "stopping", "step"),
    ):
        self.use_mlflow = use_mlflow
        self.whitelist = whitelist
        self.blacklist = blacklist

    def after_create_session(self, session, coord):
        """Log average norm and number of zeros of variables values."""
        super().after_create_session(session, coord)

        # Compute norms and num_zeros for each variable in the graph
        average_norms = defaultdict(list)
        num_zeros = defaultdict(list)
        for var in tf.global_variables():
            if self.whitelist is not None and not any(name in var.name.lower() for name in self.whitelist):
                continue
            if self.blacklist is not None and any(name in var.name.lower() for name in self.blacklist):
                continue
            value = session.run(var)
            average_norms[f"{_get_name(var)}_init_average_norm"].append(_average_norm(value))
            num_zeros[f"{_get_name(var)}_init_num_zeros"].append(_num_zeros(value))

        # Average norms and sum zeros for partitioned variables
        average_norms = {name: np.mean(values) for name, values in average_norms.items()}
        num_zeros = {name: sum(values) for name, values in num_zeros.items()}

        # Log results
        metrics = {**average_norms, **num_zeros}
        for name, value in metrics.items():
            LOGGER.info(f"{name} = {value}")
        if self.use_mlflow:
            mlflow.log_metrics(metrics)


def _get_name(var) -> str:
    """Normalize name of Tensorflow variable"""
    parts = var.name.split("/")
    filtered = "/".join([part for part in parts if not part.startswith("part")])
    return re.sub(r":\d+", "", filtered)


def _average_norm(array) -> float:
    """Compute average norm of array"""
    if len(array.shape) > 1:
        return float(np.mean(np.linalg.norm(array, axis=-1)))
    else:
        return float(np.mean(np.abs(array)))


def _num_zeros(array) -> int:
    """Compute number of zeros rows of array."""
    if len(array.shape) > 1:
        reduced = np.sum(array, axis=tuple(range(1, len(array.shape))))
        is_zero = np.equal(reduced, np.zeros_like(reduced))
    else:
        is_zero = np.equal(array, np.zeros_like(array))
    return int(np.sum(is_zero))
