"""Parse TF Records"""

from typing import Dict, List

import tensorflow as tf

from deepr.utils.field import Field
from deepr.prepros import base


class FromExample(base.Prepro):
    """Parse TF Record Sequence Example"""

    def __init__(self, fields: List[Field], num_parallel_calls: int = None, sequence: bool = None):
        super().__init__()
        self.fields = fields
        self.num_parallel_calls = num_parallel_calls
        self.sequence = sequence

    @property
    def parse_fn(self):
        """Return parse function."""

        features = {
            field.name: (
                field.feature_specs if field.is_featurizable() else tf.io.FixedLenFeature(shape=(), dtype=tf.string)
            )
            for field in self.fields
            if not field.sequence
        }
        sequence_features = {field.name: field.feature_specs for field in self.fields if field.sequence}

        def _parse_func(serialized) -> Dict[str, tf.Tensor]:
            """Parse tf.Example into dictionary of tf.Tensor"""
            if sequence_features or self.sequence:
                context, sequence = tf.io.parse_single_sequence_example(
                    serialized, context_features=features, sequence_features=sequence_features
                )
                tensors = {**context, **sequence}
            else:
                tensors = tf.io.parse_single_example(serialized, features=features)

            return {
                field.name: (
                    tensors[field.name]
                    if field.is_featurizable()
                    else tf.io.parse_tensor(tensors[field.name], out_type=field.dtype)
                )
                for field in self.fields
            }

        return _parse_func

    def apply(self, dataset: tf.data.Dataset, mode: str = None):
        return dataset.map(self.parse_fn, num_parallel_calls=self.num_parallel_calls)


TFRecordSequenceExample = FromExample  # Legacy


class ToExample(base.Prepro):
    """Convert dictionary of Tensors to tf.SequenceExample."""

    def __init__(self, fields: List[Field], num_parallel_calls: int = None, sequence: bool = None):
        super().__init__()
        self.fields = fields
        self.num_parallel_calls = num_parallel_calls
        self.sequence = sequence

    @property
    def serialize_fn(self):
        """Return parse function."""

        def _serialize_fn(*tensors):
            feature, feature_list = {}, {}
            for field, tensor in zip(self.fields, tensors):
                # Convert Eager Tensor to tf.train.Feature
                if field.is_featurizable():
                    feat = field.to_feature(tensor.numpy())
                else:
                    feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor.numpy()]))

                # Update feature and feature_list
                if isinstance(feat, tf.train.Feature):
                    feature[field.name] = feat
                elif isinstance(feat, tf.train.FeatureList):
                    feature_list[field.name] = feat
                else:
                    raise TypeError(feat)

            if feature_list or self.sequence:
                example = tf.train.SequenceExample(
                    context=tf.train.Features(feature=feature),
                    feature_lists=tf.train.FeatureLists(feature_list=feature_list),
                )
            else:
                example = tf.train.Example(features=tf.train.Features(feature=feature))
            return example.SerializeToString()

        def _tf_serialize_fn(element):
            tensors = [
                (element[field.name] if field.is_featurizable() else tf.io.serialize_tensor(element[field.name]))
                for field in self.fields
            ]
            tf_string = tf.py_function(_serialize_fn, tensors, tf.string)
            return tf.reshape(tf_string, ())

        return _tf_serialize_fn

    def apply(self, dataset: tf.data.Dataset, mode: str = None):
        return dataset.map(self.serialize_fn, num_parallel_calls=self.num_parallel_calls)
