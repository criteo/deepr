"""Saved Model Exporter"""

from typing import List
import logging

import tensorflow as tf

from deepr.utils.field import Field
from deepr.exporters import base


LOGGER = logging.getLogger(__name__)


class SavedModel(base.Exporter):
    """Saved Model Exporter

    Attributes
    ----------
    path_saved_model : str
        Path to saved_model directory
    fields : List[Field]
        List of field to build the input_receiver_fn.
    """

    def __init__(self, path_saved_model: str, fields: List[Field]):
        self.path_saved_model = path_saved_model
        self.fields = fields

    def export(self, estimator: tf.estimator.Estimator):
        features = {field.name: field.as_placeholder(batch=True) for field in self.fields}
        return estimator.export_saved_model(
            self.path_saved_model, tf.estimator.export.build_raw_serving_input_receiver_fn(features)
        )
