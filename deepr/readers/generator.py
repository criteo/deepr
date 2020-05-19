"""Reader Class for datasets using generator functions"""

from typing import Callable

import tensorflow as tf

from deepr.readers import base


class GeneratorReader(base.Reader):
    """Reader Class for datasets using generator functions

    Attributes
    ----------
    generator_fn : Callable
        Generator function, yields features, labels
    output_types : Nested structure of tf.DType
        Generator outputs dtypes
    output_shapes : Nested structure of tf.TensorShape
        Generator outputs shapes
    """

    def __init__(self, generator_fn: Callable, output_types, output_shapes=None):
        super().__init__()
        self.generator_fn = generator_fn
        self.output_types = output_types
        self.output_shapes = output_shapes

    def as_dataset(self) -> tf.data.Dataset:
        """Build a tf.data.Dataset"""
        return tf.data.Dataset.from_generator(
            self.generator_fn, output_types=self.output_types, output_shapes=self.output_shapes
        )
