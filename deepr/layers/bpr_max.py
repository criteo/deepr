"""BPT Max Loss Layer"""

import tensorflow as tf

from deepr.layers import base
from deepr.layers.reduce import Average
from deepr.layers.core import Softmax
from deepr.utils.broadcasting import make_same_shape


class BPRMax(base.Layer):
    """Vanilla BPR Max Loss Layer"""

    def __init__(self, bpr_max_regularizer=0.0, **kwargs):
        self.bpr_max_regularizer = bpr_max_regularizer
        super().__init__(n_in=2, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer

        Parameters
        ----------
        tensors : Tuple[tf.Tensor]
            - positives : shape = (batch, num_events)
            - negatives : shape = (batch, num_events, num_negatives)

        Returns
        -------
        tf.Tensor
            BPR Max loss
        """
        positives, negatives = tensors
        positives, negatives = make_same_shape([positives, negatives], broadcast=False)
        softmax_scores = Softmax()((negatives, tf.ones_like(negatives)))
        losses = -tf.math.log(tf.reduce_sum(input_tensor=tf.multiply(softmax_scores, tf.nn.sigmoid(positives - negatives)), axis=-1))
        # add bpr_max regularisation
        bpr_regularization = tf.multiply(
            tf.constant(self.bpr_max_regularizer, dtype=tf.float32),
            tf.reduce_sum(input_tensor=tf.multiply(softmax_scores, tf.square(negatives)), axis=-1),
        )
        scores = losses + bpr_regularization
        return Average()(scores, mode)


class MaskedBPRMax(base.Layer):
    """Masked BPR Max Loss Layer"""

    def __init__(self, bpr_max_regularizer=0.0, **kwargs):
        self.bpr_max_regularizer = bpr_max_regularizer
        super().__init__(n_in=4, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer
        (details: https://arxiv.org/pdf/1706.03847.pdf)

        Parameters
        ----------
        tensors : Tuple[tf.Tensor]
            - positives : shape = (batch, num_events)
            - negatives : shape = (batch, num_events, num_negatives)
            - mask : shape = (batch, num_events, num_negatives)
            - weights : shape = (batch, num_events)

        Returns
        -------
        tf.Tensor
            BPR Max loss
        """
        positives, negatives, mask, weights = tensors
        mask = tf.cast(mask, tf.float32)
        positives, negatives = make_same_shape([positives, negatives], broadcast=False)
        no_sampled_logits = tf.cast(tf.greater_equal(tf.reduce_sum(input_tensor=mask, axis=-1), 0), tf.float32)
        softmax_scores = Softmax()((negatives, mask))
        # compute bpr_max losses
        losses = -tf.multiply(
            no_sampled_logits,
            tf.math.log(
                tf.reduce_sum(input_tensor=tf.multiply(tf.multiply(softmax_scores, tf.nn.sigmoid(positives - negatives)), mask), axis=2)
                + 1e-8
            ),
        )
        # compute regularization part
        bpr_regularization = tf.multiply(
            tf.constant(self.bpr_max_regularizer, dtype=tf.float32),
            tf.reduce_sum(input_tensor=tf.multiply(tf.multiply(softmax_scores, tf.square(negatives)), mask), axis=2),
        )
        losses_with_regularization = losses + bpr_regularization
        # One loss per event, average of scores : (batch, num_events)
        # TODO: fix this line, it seems it's doing averaging twice
        # event_scores = WeightedAverage()((losses_with_regularization, mask))
        event_scores = losses_with_regularization
        # Each event contributes according to its weight
        event_weights = weights * tf.cast(tf.reduce_any(input_tensor=tf.cast(mask, tf.bool), axis=-1), dtype=tf.float32)
        event_losses = event_scores * event_weights
        return tf.math.divide_no_nan(tf.reduce_sum(input_tensor=event_losses), tf.reduce_sum(input_tensor=event_weights))
