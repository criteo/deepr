# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
"""Preprocessing"""

import tensorflow as tf

import deepr


def DefaultPrepro(batch_size, repeat_size):
    return deepr.prepros.Serial(
        deepr.prepros.TFRecordSequenceExample(
            fields=[
                deepr.Field(name="x", shape=(), dtype=tf.float32),
                deepr.Field(name="y", shape=(), dtype=tf.float32),
            ]
        ),
        deepr.prepros.Batch(batch_size=batch_size),
        deepr.prepros.Repeat(repeat_size, modes=[tf.estimator.ModeKeys.TRAIN]),
    )
