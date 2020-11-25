"""Test for utils.broadcasting"""

import pytest
import numpy as np
import tensorflow as tf

import deepr


@pytest.mark.parametrize(
    "tensors, broadcast, expected",
    [
        ([np.ones([2]), np.ones([2, 3])], True, [np.ones([2, 3]), np.ones([2, 3])]),
        ([np.ones([2]), np.ones([2, 3])], False, [np.ones([2, 1]), np.ones([2, 3])]),
    ],
)
def test_utils_make_same_shape(tensors, broadcast, expected):
    """Test make_same_shape"""
    tensors = [tf.constant(t) for t in tensors]
    got_tf = deepr.utils.make_same_shape(tensors, broadcast=broadcast)
    with tf.Session() as sess:
        got = sess.run(got_tf)
        np.testing.assert_equal(expected, got)
