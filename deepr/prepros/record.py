"""Parse TF Records"""

from typing import Dict, List, Iterable

import tensorflow as tf

from deepr.utils.field import Field
from deepr.prepros import core


class FromExample(core.Map):
    """Parse TF Record Sequence Example"""

    def __init__(
        self,
        fields: List[Field],
        sequence: bool = None,
        modes: Iterable[str] = None,
        num_parallel_calls: int = None,
        batched: bool = False,
    ):
        self.fields = fields
        self.sequence = sequence
        self.batched = batched

        features = {
            field.name: (
                field.feature_specs if field.is_featurizable() else tf.io.FixedLenFeature(shape=(), dtype=tf.string)
            )
            for field in self.fields
            if not field.sequence
        }
        sequence_features = {field.name: field.feature_specs for field in self.fields if field.sequence}

        def _map_func(serialized) -> Dict[str, tf.Tensor]:
            """Parse tf.Example into dictionary of tf.Tensor"""
            if sequence_features or self.sequence:
                if self.batched:
                    context, sequence, _ = tf.io.parse_sequence_example(
                        serialized, context_features=features, sequence_features=sequence_features
                    )
                else:
                    context, sequence = tf.io.parse_single_sequence_example(
                        serialized, context_features=features, sequence_features=sequence_features
                    )
                tensors = {**context, **sequence}
            else:
                if self.batched:
                    tensors = tf.io.parse_example(serialized, features=features)
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

        super().__init__(
            map_func=_map_func, on_dict=False, update=False, num_parallel_calls=num_parallel_calls, modes=modes
        )


TFRecordSequenceExample = FromExample  # Legacy


class ToExample(core.Map):
    """Convert dictionary of Tensors to tf.SequenceExample."""

    def __init__(
        self, fields: List[Field], sequence: bool = None, modes: Iterable[str] = None, num_parallel_calls: int = None
    ):
        self.fields = fields
        self.sequence = sequence

        def _map_func_np(*tensors):
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

        def _map_func_tf(element):
            tensors = [
                (element[field.name] if field.is_featurizable() else tf.io.serialize_tensor(element[field.name]))
                for field in self.fields
            ]
            tf_string = tf.py_function(_map_func_np, tensors, tf.string)
            return tf.reshape(tf_string, ())

        super().__init__(
            map_func=_map_func_tf, on_dict=False, update=False, num_parallel_calls=num_parallel_calls, modes=modes
        )
