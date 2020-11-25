# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""Squared L2 loss"""

import logging

import tensorflow as tf

import deepr


LOGGER = logging.getLogger(__name__)


@deepr.layers.layer(inputs=("y", "y_pred"), outputs="loss")
def SquaredL2(tensors):
    x, y = tensors
    return tf.reduce_sum((x - y) ** 2)
