"""Accuracy metrics."""

from typing import Dict, Tuple

import tensorflow as tf

from deepr.metrics import base


class Accuracy(base.Metric):
    """Accuracy."""

    def __init__(self, gold: str, pred: str, name: str):
        self.gold = gold
        self.pred = pred
        self.name = name

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, Tuple]:
        return {self.name: tf.metrics.accuracy(labels=tensors[self.gold], predictions=tensors[self.pred])}


class AccuracyAtK(base.Metric):
    """Accuracy at k."""

    def __init__(self, gold: str, logits: str, name: str, k: int):
        self.gold = gold
        self.logits = logits
        self.name = name
        self.k = k

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, Tuple]:
        in_top_k = tf.math.in_top_k(targets=tensors[self.gold], predictions=tensors[self.logits], k=self.k)
        prop_in_top_k = tf.reduce_mean(tf.cast(in_top_k, tf.float32))
        return {self.name: tf.metrics.mean(prop_in_top_k)}
