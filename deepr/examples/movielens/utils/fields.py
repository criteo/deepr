"""Fields for MovieLens."""

import tensorflow as tf

import deepr as dpr


UID = dpr.Field(name="uid", shape=(), dtype=tf.int64)

INPUT_POSITIVES = dpr.Field(name="inputPositives", shape=(None,), dtype=tf.int64)

TARGET_POSITIVES = dpr.Field(name="targetPositives", shape=(None,), dtype=tf.int64)

TARGET_NEGATIVES = dpr.Field(name="targetNegatives", shape=(None, None), dtype=tf.int64)

INPUT_MASK = dpr.Field(name="inputMask", dtype=tf.bool, shape=(None,), default=False)

TARGET_MASK = dpr.Field(name="targetMask", dtype=tf.bool, shape=(None,), default=False)


def INPUT_POSITIVES_ONE_HOT(vocab_size):
    # pylint: disable=invalid-name
    return dpr.Field(name="inputPositivesOneHot", shape=(vocab_size,), dtype=tf.int64)


def TARGET_POSITIVES_ONE_HOT(vocab_size):
    # pylint: disable=invalid-name
    return dpr.Field(name="targetPositivesOneHot", shape=(vocab_size,), dtype=tf.int64)
