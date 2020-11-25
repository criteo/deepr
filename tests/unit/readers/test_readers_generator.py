"""Test readers.generator"""

import tensorflow as tf

import deepr


def test_readers_generator():
    """Test GeneratorReader"""

    def generator_fn():
        for idx in range(5):
            yield idx, 2 * idx

    reader = deepr.readers.GeneratorReader(generator_fn, output_types=(tf.int32, tf.int32), output_shapes=((), ()))
    assert list(generator_fn()) == list(reader)
