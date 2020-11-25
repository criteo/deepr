"""Tests for jobs.save_dataset"""

import numpy as np
import tensorflow as tf

import deepr


def test_jobs_save_dataset(tmpdir):
    """Test SaveDataset"""
    path = str(tmpdir.join("data"))
    field = deepr.Field(name="x", shape=(2, 2), dtype=tf.int64)

    # Define dataset
    def _gen():
        for idx in range(5):
            yield {"x": np.reshape(np.arange(4) * idx, (2, 2))}

    # Run SaveDataset job
    input_fn = deepr.readers.GeneratorReader(_gen, output_types={"x": field.dtype}, output_shapes={"x": field.shape})
    job = deepr.jobs.SaveDataset(input_fn=input_fn, path=path, fields=[field], chunk_size=2, secs=1)
    job.run()

    # Read dataset
    reader = deepr.readers.TFRecordReader(path=path, shuffle=False, num_parallel_reads=None, num_parallel_calls=None)
    prepro_fn = deepr.prepros.FromExample([field])
    idx = 0
    for idx, (got, expected) in enumerate(zip(deepr.readers.from_dataset(prepro_fn(reader())), _gen())):
        np.testing.assert_equal(got, expected)
    assert idx == 4
