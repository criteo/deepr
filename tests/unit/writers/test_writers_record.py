"""Tests for writers.record."""

import numpy as np
import tensorflow as tf
import pytest

import deepr


@pytest.mark.parametrize("shape", [[1], [2], [2, 3], [None, 3], [2, 3, 4], [None, 3, 4]])
@pytest.mark.parametrize("dtype", [tf.int64, tf.float32])
@pytest.mark.parametrize("chunk_size", [None, 2])
def test_writers_record_simple(tmpdir, shape, dtype, chunk_size):
    """Simple test for record writer."""
    path = str(tmpdir.join("data.tfrecord.gz")) if not chunk_size else str(tmpdir.join("data"))
    size = 1
    static_shape = [s if s is not None else 2 for s in shape]
    for s in static_shape:
        size *= s

    # Define dataset
    def _gen():
        for idx in range(5):
            yield {"x": np.reshape(np.arange(size) * idx, static_shape)}

    dataset = tf.data.Dataset.from_generator(_gen, output_types={"x": dtype}, output_shapes={"x": shape})

    # Write dataset
    field = deepr.Field(name="x", shape=shape, dtype=dtype)
    prepro_fn = deepr.prepros.ToExample([field])
    writer = deepr.writers.TFRecordWriter(path=path, chunk_size=chunk_size)
    writer.write(prepro_fn(dataset))

    # Read dataset
    reader = deepr.readers.TFRecordReader(path=path, shuffle=False, num_parallel_reads=None, num_parallel_calls=None)
    prepro_fn = deepr.prepros.FromExample([field])
    idx = 0
    for idx, (got, expected) in enumerate(zip(deepr.readers.from_dataset(prepro_fn(reader())), _gen())):
        np.testing.assert_equal(got, expected)
    assert idx == 4
