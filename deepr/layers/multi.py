# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""Negative Multinomial Log Likelihood."""

import tensorflow as tf

from deepr.layers import base


class MultiLogLikelihood(base.Layer):
    """Negative Multinomial Log Likelihood."""

    def __init__(self, **kwargs):
        super().__init__(n_in=2, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Multinomial Log Likelihood

        Parameters
        ----------
        tensors : Tuple[tf.Tensor]
            - logits : shape = (batch, num_classes), tf.float32
            - classes : shape = (batch, num_classes), tf.int64 as a
            one-hot vector

        Returns
        -------
        tf.Tensor
            Negative Multinomial Log Likelihood, scalar
        """
        logits, classes = tensors
        log_softmax = tf.nn.log_softmax(logits)
        return -tf.reduce_mean(tf.reduce_sum(log_softmax * tf.cast(classes, tf.float32), axis=-1))
