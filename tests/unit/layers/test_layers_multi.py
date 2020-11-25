# pylint: disable=invalid-name
"""Tests for layers.multi"""

import numpy as np
import tensorflow as tf

import deepr


def test_layers_multi():
    """Compare MultiLogLikelihood output with a NumPy implementation."""
    batch_size = 2
    num_classes = 10

    log_likelihood = deepr.layers.MultiLogLikelihood(inputs=("logits", "classes"), outputs="log_likelihood")

    np.random.seed(2020)

    inputs = {
        "logits": np.random.random([batch_size, num_classes]),
        "classes": np.random.randint(2, size=[batch_size, num_classes]),
    }

    # Compute Multinomial Log Likelihood with NumPy
    expected = 0
    for batch in range(batch_size):
        Z = np.sum(np.exp(inputs["logits"][batch]))
        for idx in range(num_classes):
            expected -= inputs["classes"][batch, idx] * (inputs["logits"][batch, idx] - np.log(Z))
    expected /= batch_size

    # Compare with Tensorflow Value
    inputs = {key: tf.constant(val, dtype=tf.int64 if key == "classes" else tf.float32) for key, val in inputs.items()}
    result = log_likelihood(inputs)
    with tf.Session() as sess:
        result_eval = sess.run(result["log_likelihood"])
        np.testing.assert_allclose(result_eval, expected)
