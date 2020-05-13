"""Summary Saver Hook"""

from typing import List, Dict

import tensorflow as tf

from deepr.hooks.base import TensorHookFactory


class SummarySaverHookFactory(TensorHookFactory):
    """Summary Saver Hook"""

    def __init__(self, tensors: List[str] = None, **kwargs):
        self.tensors = tensors
        self._kwargs = kwargs

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> tf.estimator.SessionRunHook:
        # Extract relevant tensors
        if self.tensors is None:
            tensors = {key: tensor for key, tensor in tensors.items() if len(tensor.shape) == 0}
        else:
            tensors = {name: tensors[name] for name in self.tensors}

        # Define summaries
        for name, tensor in tensors.items():
            tf.summary.scalar(name, tensor)

        return tf.train.SummarySaverHook(summary_op=tf.summary.merge_all(), **self._kwargs)
