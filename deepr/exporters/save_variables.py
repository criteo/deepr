"""Save Variables Exporter"""

import logging
from typing import List

import tensorflow as tf
import pandas as pd

from deepr.exporters import base
from deepr.io.parquet import ParquetDataset
from deepr.io.path import Path


LOGGER = logging.getLogger(__name__)


class SaveVariables(base.Exporter):
    """Save Variables as Parquet, supports chunking.

    Attributes
    ----------
    path_variables : str
        Path to export directory
    variable_names : str
        Name of variables from the Tensorflow Graph.
    chunk_size : int
        Number of elements per checkpoint
    compression : str
        Type of compression, default to "snappy"
    """

    def __init__(
        self, path_variables: str, variable_names: List[str], chunk_size: int = 100_000, compression: str = "snappy"
    ):
        self.path_variables = path_variables
        self.variable_names = variable_names
        self.chunk_size = chunk_size
        self.compression = compression

    def export(self, estimator: tf.estimator.Estimator):
        for variable_name in self.variable_names:
            variable_export_dir = Path(self.path_variables, variable_name)
            LOGGER.info(f"Saving variable {variable_name} to {variable_export_dir}")
            with ParquetDataset(variable_export_dir).open() as ds:
                variable_value = estimator.get_variable_value(variable_name)
                ds.write_pandas(pd.DataFrame(variable_value), compression=self.compression, chunk_size=self.chunk_size)
