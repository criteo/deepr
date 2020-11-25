"""Tests for layers.top_one_max"""

import numpy as np
import tensorflow as tf

import deepr


def test_layers_top_one_max():
    """Compare TopOne Max `Layer` output with a dummy NumPy implementation"""
    batch_size = 2
    num_target = 4
    num_negatives = 8
    dim = 16

    bpr_max_layer = deepr.layers.DAG(
        deepr.layers.Select(inputs=("preds", "positives", "negatives", "mask", "weights")),
        deepr.layers.DotProduct(inputs=("preds", "positives"), outputs="positive_logits"),
        deepr.layers.DotProduct(inputs=("preds", "negatives"), outputs="negative_logits"),
        deepr.layers.MaskedTopOneMax(inputs=("positive_logits", "negative_logits", "mask", "weights"), outputs="loss"),
    )

    np.random.seed(2020)

    inputs = {
        "preds": np.random.random([batch_size, dim]),
        "positives": np.random.random([batch_size, num_target, dim]),
        "negatives": np.random.random([batch_size, num_target, num_negatives, dim]),
        "mask": np.random.randint(0, 2, size=[batch_size, num_target, num_negatives]),
        "weights": np.random.random([batch_size, num_target]),
    }

    # Compute TopOne Max loss naively
    event_scores = 0
    event_weights = 0
    for batch in range(batch_size):
        for target in range(num_target):
            # Compute positive logits
            pos_product = 0
            for d in range(dim):
                pos_product += inputs["preds"][batch, d] * inputs["positives"][batch, target, d]

            # Compute negative logits
            neg_products = []
            for negative in range(num_negatives):
                # Compute negative logits
                neg_product = 0
                for d in range(dim):
                    neg_product += inputs["preds"][batch, d] * inputs["negatives"][batch, target, negative, d]
                neg_products.append(neg_product)

            # Compute softmaxes for neg_products:
            negatives_tensor = np.array(neg_products)
            negatives_mask = inputs["mask"][batch, target]
            exp_negatives = np.exp(negatives_tensor - np.max(negatives_tensor * negatives_mask))
            sum_exp_negatives = np.sum(exp_negatives * negatives_mask)
            softmaxes = exp_negatives / sum_exp_negatives if sum_exp_negatives != 0 else np.zeros_like(exp_negatives)

            # For each negative, compute score contribution
            scores = 0
            negative_mask = 0
            for negative in range(num_negatives):
                m = inputs["mask"][batch, target, negative]
                neg_product = neg_products[negative]
                scores += (
                    softmaxes[negative]
                    * (1 / (1 + np.exp(-(neg_product - pos_product))) + 1 / (1 + np.exp(-(neg_product ** 2))))
                    * m
                )
                negative_mask += m

            scores = scores / negative_mask if negative_mask else 0

            # Compute event score and weight
            event_weight = inputs["weights"][batch, target] if negative_mask > 0 else 0
            event_scores += scores * event_weight
            event_weights += event_weight

    expected = event_scores / event_weights if event_weights > 0 else 0

    # Compare with Tensorflow Value
    inputs = {key: tf.constant(val, dtype=tf.bool if key == "mask" else tf.float32) for key, val in inputs.items()}
    result = bpr_max_layer(inputs)
    with tf.Session() as sess:
        result_eval = sess.run(result["loss"])
        np.testing.assert_allclose(result_eval, expected, 1e-4)
