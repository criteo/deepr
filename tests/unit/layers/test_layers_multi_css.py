# pylint: disable=invalid-name
"""Tests for layers.multi_css"""

import numpy as np
import tensorflow as tf

import deepr


def test_layers_multi_css():
    """Compare MultiLogLikelihoodCSS with NumPy implementation."""
    batch_size = 2
    num_positives = 4
    num_negatives = 8
    vocab_size = 10

    log_likelihood = deepr.layers.MultiLogLikelihoodCSS(
        inputs=("positive_logits", "negative_logits", "positive_mask", "negative_mask"),
        outputs="log_likelihood",
        vocab_size=vocab_size,
    )

    np.random.seed(2020)

    inputs = {
        "positive_logits": np.random.random([batch_size, num_positives]),
        "positive_mask": np.random.randint(2, size=[batch_size, num_positives]),
        "negative_logits": np.random.random([batch_size, num_positives, num_negatives]),
        "negative_mask": np.random.randint(2, size=[batch_size, num_positives, num_negatives]),
    }

    # Compute Multinomial Log Likelihood with NumPy
    expected = 0
    for batch in range(batch_size):
        for positive in range(num_positives):
            logit = inputs["positive_logits"][batch, positive]
            u = np.exp(logit)
            Z_c = np.sum(np.exp(inputs["negative_logits"][batch, positive]) * inputs["negative_mask"][batch, positive])
            N = float(np.sum(inputs["negative_mask"][batch, positive]))
            log_Z = np.log(u + (vocab_size - 1) * Z_c / N) if N != 0 else np.log(u)
            expected += -logit + log_Z if inputs["positive_mask"][batch, positive] else 0
    expected /= batch_size

    # Compare with Tensorflow Value
    inputs = {key: tf.constant(val, dtype=tf.bool if "mask" in key else tf.float32) for key, val in inputs.items()}
    result = log_likelihood(inputs)
    with tf.Session() as sess:
        result_eval = sess.run(result["log_likelihood"])
        np.testing.assert_allclose(result_eval, expected)
