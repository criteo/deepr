"""Recall metrics."""

from typing import Dict, Tuple

import tensorflow as tf

import deepr as dpr


class RecallAtK(dpr.metrics.Metric):
    """Recall at k.

    Attributes
    ----------
    name : str
        Name of the metric
    logits : str
        Tensor with logits, shape = (batch, num_classes)
    targets : str
        Tensor with indices, shape = (batch, ..., num_target)
    k : int
        Top k logits for recall
    inputs : str, optional
        One-hot tensor of "masked" classes, shape = (batch, num_casses)
    """

    def __init__(self, name: str, logits: str, targets: str, k: int, inputs: str = None):
        self.name = name
        self.logits = logits
        self.targets = targets
        self.inputs = inputs
        self.k = k

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, Tuple]:
        # Retrieve tensors
        logits = tensors[self.logits]  # (batch, num_classes)
        targets = tensors[self.targets]  # (batch, ..., num_targets)
        ndims = len(targets.shape) - 1

        # Set logits of inputs to -inf
        if self.inputs is not None:
            inputs = tensors[self.inputs]  # (batch, num_classes)
            logits = logits + tf.cast(inputs, tf.float32) * tf.float32.min

        # Retrieve top k predictions, shape = (batch, k)
        _, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        # shape = (batch, ..., 1, k)
        for _ in range(ndims):
            indices = tf.expand_dims(indices, axis=-2)
        # shape = (batch, ..., num_targets, k)
        equal_topk = tf.equal(tf.cast(indices, tf.int64), tf.expand_dims(targets, axis=-1))

        # Compute number of items in top k
        num_in_topk = tf.reduce_sum(tf.reduce_sum(tf.cast(equal_topk, tf.int64), axis=-1), axis=-1)
        num_targets = tf.reduce_sum(tf.cast(tf.not_equal(targets, -1), tf.int64), axis=-1)
        num_targets = tf.math.minimum(num_targets, self.k)
        recall_at_k = tf.div_no_nan(tf.cast(num_in_topk, tf.float32), tf.cast(num_targets, tf.float32))
        return {self.name: tf.metrics.mean(recall_at_k)}
