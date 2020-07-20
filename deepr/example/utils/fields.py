"""Fields."""

import tensorflow as tf

import deepr as dpr


UID = dpr.Field(name="uid", shape=(), dtype=tf.string)

INPUT_POSITIVES = dpr.Field(name="inputPositives", shape=(None,), dtype=tf.int64)

TARGET_POSITIVES = dpr.Field(name="targetPositives", shape=(None,), dtype=tf.int64)

TARGET_NEGATIVES = dpr.Field(name="targetNegatives", shape=(None, None), dtype=tf.int64)

FIELDS_MOVIELENS = [UID, INPUT_POSITIVES, TARGET_POSITIVES, TARGET_NEGATIVES]
