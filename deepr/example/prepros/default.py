# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
"""Preprocessing"""

import tensorflow as tf

import deepr as dpr
import deepr.prepros as dprp


def DefaultPrepro(batch_size, repeat_size):
    return dpr.prepros.Serial(
        dprp.TFRecordSequenceExample(
            fields=[dpr.Field(name="x", shape=(), dtype=tf.float32), dpr.Field(name="y", shape=(), dtype=tf.float32)]
        ),
        dpr.prepros.Batch(batch_size=batch_size),
        dpr.prepros.Repeat(repeat_size, modes=[tf.estimator.ModeKeys.TRAIN]),
    )
