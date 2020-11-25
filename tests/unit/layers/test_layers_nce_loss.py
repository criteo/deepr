"""Tests for layers.bpr"""

import numpy as np
import tensorflow as tf

import deepr


def test_layers_nce_loss():
    """Compare NegativeSampling `Layer` output with a dummy NumPy implementation"""
    batch_size = 2
    num_target = 4
    num_negatives = 8
    dim = 16

    nce_layer = deepr.layers.DAG(
        deepr.layers.Select(inputs=("preds", "positives", "negatives", "mask", "weights")),
        deepr.layers.DotProduct(inputs=("preds", "positives"), outputs="positive_logits"),
        deepr.layers.DotProduct(inputs=("preds", "negatives"), outputs="negative_logits"),
        deepr.layers.MaskedNegativeSampling(
            inputs=("positive_logits", "negative_logits", "mask", "weights"), outputs="loss"
        ),
    )

    np.random.seed(2020)

    inputs = {
        "preds": np.random.random([batch_size, dim]),
        "positives": np.random.random([batch_size, num_target, dim]),
        "negatives": np.random.random([batch_size, num_target, num_negatives, dim]),
        "mask": np.random.randint(0, 2, size=[batch_size, num_target, num_negatives]),
        "weights": np.random.random([batch_size, num_target]),
    }

    # Compute negative sampling loss naively
    event_scores = 0
    event_weights = 0
    for batch in range(batch_size):
        for target in range(num_target):
            negative_scores = 0
            negative_mask = 0

            # Compute the score contribution for positive
            pos_product = 0
            for d in range(dim):
                pos_product += inputs["preds"][batch, d] * inputs["positives"][batch, target, d]
            # Based on https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits:
            positive_score = np.max(pos_product, 0) - pos_product * 1 + np.log(1 + np.exp(-abs(pos_product)))

            # For each negative, compute score contribution
            for negative in range(num_negatives):
                neg_product = 0
                for d in range(dim):
                    neg_product += inputs["preds"][batch, d] * inputs["negatives"][batch, target, negative, d]
                m = inputs["mask"][batch, target, negative]
                # Based on https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits:
                negative_scores += (
                    np.max(neg_product, 0) - neg_product * 0 + np.log(1 + np.exp(-np.abs(neg_product)))
                ) * m
                negative_mask += m

            # Compute event score and weight
            event_weight = inputs["weights"][batch, target] if negative_mask > 0 else 0
            event_score = negative_scores / negative_mask if negative_mask > 0 else 0
            event_scores += (event_score + positive_score) * event_weight
            event_weights += event_weight

    expected = event_scores / event_weights if event_weights > 0 else 0

    # Compare with Tensorflow Value
    inputs = {key: tf.constant(val, dtype=tf.bool if key == "mask" else tf.float32) for key, val in inputs.items()}
    result = nce_layer(inputs)
    with tf.Session() as sess:
        result_eval = sess.run(result["loss"])
        np.testing.assert_allclose(result_eval, expected, 1e-4)
