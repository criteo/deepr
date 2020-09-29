"""NDCG metrics."""

from typing import Dict, Tuple

import tensorflow as tf

import deepr as dpr


class NDCGAtK(dpr.metrics.Metric):
    """Normalized Discounted Cumulative Gain at K

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
        logits = tensors[self.logits]
        targets = tensors[self.targets]
        ndims = len(targets.shape) - 1

        # Set logits of inputs to -inf
        if self.inputs is not None:
            inputs = tensors[self.inputs]
            logits = logits + tf.cast(inputs, tf.float32) * tf.float32.min

        # Retrieve top k predictions
        _, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        for _ in range(ndims):
            indices = tf.expand_dims(indices, axis=-2)
        equal_topk = tf.equal(tf.cast(indices, tf.int64), tf.expand_dims(targets, axis=-1))

        # Discounted cumulative gain
        pos_in_target = tf.reduce_sum(tf.cast(equal_topk, tf.float32), axis=-2)
        discount = tf.math.log(2.0) / tf.math.log(tf.range(2, self.k + 2, dtype=tf.float32))
        for _ in range(ndims + 1):
            discount = tf.expand_dims(discount, axis=0)
        dcg = tf.reduce_sum(discount * pos_in_target, axis=-1)

        # Ideal discounted cumulative gain
        num_targets = tf.reduce_sum(tf.cast(tf.not_equal(targets, -1), tf.int64), axis=-1)
        num_targets = tf.math.minimum(num_targets, self.k)
        all_in_target = tf.cast(tf.sequence_mask(num_targets, maxlen=self.k), tf.float32)
        idcg = tf.reduce_sum(discount * all_in_target, axis=-1)

        # Normalized DCG
        ndcg = tf.div_no_nan(dcg, idcg)
        return {self.name: tf.metrics.mean(ndcg)}
