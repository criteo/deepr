"""Base Hooks Factories

Some TensorFlow hooks cannot be defined before runtime. For example, a
:class:`~TensorLoggingHook` requires `tensors` to be initialized.

To resolve this issue, we provide abstractions for Hooks factories, that
allow you to parametrize the creation of hooks that will be created at
runtime.

See the :class:`~LoggingTensorHookFactory` for instance.
"""

from abc import ABC, abstractmethod
from typing import Dict

import tensorflow as tf


class TensorHookFactory(ABC):
    """Tensor Hook Factory"""

    @abstractmethod
    def __call__(self, tensors: Dict[str, tf.Tensor]) -> tf.estimator.SessionRunHook:
        raise NotImplementedError()


class EstimatorHookFactory(ABC):
    """Estimator Hook Factory"""

    @abstractmethod
    def __call__(self, estimator: tf.estimator.Estimator) -> tf.estimator.SessionRunHook:
        raise NotImplementedError()
