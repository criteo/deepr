"""Parse TF Records"""

from typing import Dict, List

import tensorflow as tf

from deepr.utils.field import Field
from deepr.prepros import base


class TFRecordSequenceExample(base.Prepro):
    """Parse TF Record Sequence Example"""

    def __init__(self, fields: List[Field], num_parallel_calls: int = None):
        super().__init__()
        self.fields = fields
        self.num_parallel_calls = num_parallel_calls

    @property
    def parse_fn(self):
        """Return parse function"""

        def _parse_func(element) -> Dict[str, tf.Tensor]:
            """Parse tf.Example into dictionary of tf.Tensor"""
            fields_context = [field for field in self.fields if len(field.shape) < 2]
            fields_sequence = [field for field in self.fields if len(field.shape) >= 2]
            context, sequence = tf.parse_single_sequence_example(
                element,
                context_features={field.name: field.as_feature() for field in fields_context},
                sequence_features={field.name: field.as_feature() for field in fields_sequence},
            )
            return {**context, **sequence}

        return _parse_func

    def apply(self, dataset: tf.data.Dataset, mode: str = None):
        return dataset.map(self.parse_fn, num_parallel_calls=self.num_parallel_calls)
