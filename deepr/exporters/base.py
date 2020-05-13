"""Base class for Exporters"""

from abc import ABC, abstractmethod

import tensorflow as tf


class Exporter(ABC):
    """Base class for Exporters"""

    def __call__(self, estimator: tf.estimator.Estimator):
        """Alias for export"""
        return self.export(estimator)

    @abstractmethod
    def export(self, estimator: tf.estimator.Estimator):
        raise NotImplementedError()
