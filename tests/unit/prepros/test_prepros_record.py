# pylint: disable=redefined-outer-name
"""Tests for prepros.record"""

import pytest
import tensorflow as tf

import deepr as dpr


@pytest.fixture
def dummy_tfrecord(tmpdir) -> str:
    """Writes a dummy tf record file"""
    # Create tf.SequenceExample
    context = {"a": dpr.readers.bytes_feature([b"0"])}
    feature_list = {"b": dpr.readers.int64_feature_list([[0, 1], [2, 3]])}
    sample = tf.train.SequenceExample(
        context=tf.train.Features(feature=context), feature_lists=tf.train.FeatureLists(feature_list=feature_list)
    )

    # Write TFRecord file
    path = str(tmpdir.join("dummy.tfrecord"))
    with tf.python_io.TFRecordWriter(path) as writer:
        writer.write(sample.SerializeToString())

    return path


def test_prepro_record_sequence_example(dummy_tfrecord: str):
    """Test TFRecordSequenceReader"""
    fields = [dpr.Field(name="a", shape=(), dtype=tf.string), dpr.Field(name="b", shape=(None, 2), dtype=tf.int64)]
    reader = dpr.readers.TFRecordReader([dummy_tfrecord])
    prepro = dpr.prepros.TFRecordSequenceExample(fields=fields)
    dataset = reader.as_dataset()
    dataset = prepro(dataset)
    elements = list(dpr.readers.from_dataset(dataset))
    assert len(elements) == 1
    assert elements[0].keys() == {"a", "b"}
