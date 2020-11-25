"""Fields for MovieLens."""

import tensorflow as tf

import deepr


UID = deepr.Field(name="uid", shape=(), dtype=tf.int64)

INPUT_POSITIVES = deepr.Field(name="inputPositives", shape=(None,), dtype=tf.int64)

TARGET_POSITIVES = deepr.Field(name="targetPositives", shape=(None,), dtype=tf.int64)

TARGET_NEGATIVES = deepr.Field(name="targetNegatives", shape=(None, None), dtype=tf.int64)

INPUT_MASK = deepr.Field(name="inputMask", dtype=tf.bool, shape=(None,), default=False)

TARGET_MASK = deepr.Field(name="targetMask", dtype=tf.bool, shape=(None,), default=False)


def INPUT_POSITIVES_ONE_HOT(vocab_size):
    # pylint: disable=invalid-name
    return deepr.Field(name="inputPositivesOneHot", shape=(vocab_size,), dtype=tf.int64)


def TARGET_POSITIVES_ONE_HOT(vocab_size):
    # pylint: disable=invalid-name
    return deepr.Field(name="targetPositivesOneHot", shape=(vocab_size,), dtype=tf.int64)
