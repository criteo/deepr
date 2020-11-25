# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
"""Inference Preprocessor"""

import tensorflow as tf

import deepr


def InferencePrepro(batch_size, count, inputs: str = "inputs/x"):
    return deepr.prepros.Serial(
        deepr.prepros.TFRecordSequenceExample(
            fields=[
                deepr.Field(name="x", shape=(), dtype=tf.float32),
                deepr.Field(name="y", shape=(), dtype=tf.float32),
            ]
        ),
        deepr.prepros.Map(deepr.layers.Select(inputs="x", outputs=inputs), update=False),
        deepr.prepros.Batch(batch_size=batch_size),
        deepr.prepros.Take(count=count),
    )
