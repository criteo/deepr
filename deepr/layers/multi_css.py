# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""Multinomial Log Likelihood with Complementarity Sum Sampling."""

import tensorflow as tf

from deepr.layers import base


class MultiLogLikelihoodCSS(base.Layer):
    """Multinomial Log Likelihood with Complementarity Sum Sampling.

    http://proceedings.mlr.press/v54/botev17a/botev17a.pdf
    """

    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(n_in=4, n_out=1, **kwargs)
        self.vocab_size = vocab_size

    def forward(self, tensors, mode: str = None):
        """Multinomial Log Likelihood with Complementarity Sum Sampling.

        Parameters
        ----------
        tensors : Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
            - positive_logits: (batch, num_positives)
            - negative_logits: (batch, num_positives or 1, num_negatives)
            - positive_mask: same shape as positive logits
            - negative_mask: same shape as negative logits

        Returns
        -------
        tf.Tensor
            Multinomial Log-Likelihood with Complementarity Sampling
        """
        positive_logits, negative_logits, positive_mask, negative_mask = tensors

        # Remove -max for numerical stability
        max_negative_logits = tf.reduce_max(negative_logits, axis=-1)
        max_positive_logits = tf.maximum(positive_logits, max_negative_logits)
        max_logits = tf.reduce_max(max_positive_logits, axis=-1)
        positive_logits -= tf.expand_dims(max_logits, axis=-1)
        negative_logits -= tf.expand_dims(tf.expand_dims(max_logits, axis=-1), axis=-1)

        # Exponential of positive and negative logits
        u_p = tf.exp(positive_logits)
        u_ns = tf.exp(negative_logits)
        u_ns *= tf.cast(negative_mask, tf.float32)

        # Approximate partition function using negatives
        Z_c = tf.reduce_sum(u_ns, axis=-1)
        num_negatives = tf.reduce_sum(tf.cast(negative_mask, tf.float32), axis=-1)
        Z = u_p + (self.vocab_size - 1) * tf.div_no_nan(Z_c, num_negatives)

        # Compute Approximate Log Softmax
        log_p = positive_logits - tf.log(Z)
        log_p *= tf.cast(positive_mask, tf.float32)

        # Sum (Multinomial Log Likelihood) over positives
        multi_likelihood = tf.reduce_sum(log_p, axis=-1)

        return -tf.reduce_mean(multi_likelihood)
