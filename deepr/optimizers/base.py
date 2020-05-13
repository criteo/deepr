"""Interface for Optimizers"""

from abc import ABC
from typing import Dict

import tensorflow as tf


class Optimizer(ABC):
    """Interface for Optimizers"""

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict:
        """Return train_op in a dictionary"""
        raise NotImplementedError()
