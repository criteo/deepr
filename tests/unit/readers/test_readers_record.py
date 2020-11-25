# pylint: disable=redefined-outer-name
"""Test readers.record"""

import tensorflow as tf
import pytest

import deepr


@pytest.fixture
def dummy_tfrecord(tmpdir) -> str:
    """Writes a dummy tf record file"""
    # Create tf.SequenceExample
    context = {"a": deepr.readers.bytes_feature([b"0"])}
    feature_list = {"b": deepr.readers.int64_feature_list([[0, 1], [2, 3]])}
    sample = tf.train.SequenceExample(
        context=tf.train.Features(feature=context), feature_lists=tf.train.FeatureLists(feature_list=feature_list)
    )

    # Write TFRecord file
    path = str(tmpdir.join("dummy.tfrecord"))
    with tf.io.TFRecordWriter(path) as writer:
        writer.write(sample.SerializeToString())

    return path


def test_readers_record(dummy_tfrecord: str):
    """Test TFRecordReader"""
    reader = deepr.readers.TFRecordReader(dummy_tfrecord)
    assert len(list(reader)) == 1
