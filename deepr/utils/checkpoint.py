"""Checkpoint utilities"""

# TODO: cleanup

import logging
from typing import Dict, Optional
import os
import gc

import numpy as np
import tensorflow as tf


LOGGER = logging.getLogger(__name__)


NUMPY_TO_TF_DTYPES = {
    np.dtype("float32"): tf.float32,
    np.dtype("float64"): tf.float64,
    np.dtype("int32"): tf.int32,
    np.dtype("int64"): tf.int64,
}


def save_variables_in_ckpt(path: str, variables: Dict[str, Optional[np.ndarray]], num_shards_embeddings: int = 1):
    """Save variables in checkpoint"""
    variables_tf = {}
    tf.logging.info("Creating TF Variables from %s", list(variables.keys()))
    variables_to_save = []
    # We set a variable scope here with AUTO_REUSE because of the unit tests that reuse variables
    with tf.Graph().as_default():
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            for name, value in variables.items():
                if value is not None:
                    # We create placeholders and assignement ops here to get over the 2GB graphdef limit
                    # And not put the variable values directly in the graph
                    var = tf.get_variable(
                        name=name,
                        shape=value.shape,
                        dtype=NUMPY_TO_TF_DTYPES[value.dtype],
                        partitioner=tf.fixed_size_partitioner(num_shards=num_shards_embeddings, axis=0),
                    )
                    placeholder = tf.placeholder(dtype=value.dtype, shape=value.shape)
                    assign = var.assign(placeholder)
                    variables_tf[name] = (assign, placeholder, value)
                    variables_to_save.append(var)

        saver = tf.train.Saver(variables_to_save)
        full_path = os.path.join(path, "initial_variables.ckpt")
        tf.logging.info("Saving TF Variables %s to %s", list(variables.keys()), full_path)

        # There is no reason to use GPU for getting variable values.
        with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as sess:
            sess.run(tf.global_variables_initializer())
            for name, (assign, placeholder, value) in variables_tf.items():
                tf.logging.info("Assigning variable %s with shape %s", name, value.shape)
                sess.run(assign, {placeholder: value})
            saved_path = saver.save(sess, full_path)
            tf.logging.info("Saved TF Variables %s to %s", list(variables.keys()), saved_path)
    gc.collect()
    return saved_path
