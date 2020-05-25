# pylint: disable=no-value-for-parameter,unexpected-keyword-arg
"""Inference Preprocessor"""

import tensorflow as tf

import deepr as dpr


@dpr.prepros.prepro
def InferencePrepro(batch_size, count):
    return dpr.prepros.Serial(
        dpr.prepros.TFRecordSequenceExample(
            fields=[dpr.Field(name="x", shape=(), dtype=tf.float32), dpr.Field(name="y", shape=(), dtype=tf.float32)]
        ),
        dpr.prepros.Map(dpr.layers.Select(inputs="x", outputs="inputs/x"), update=False),
        dpr.prepros.Batch(batch_size=batch_size),
        dpr.prepros.Take(count=count),
    )
