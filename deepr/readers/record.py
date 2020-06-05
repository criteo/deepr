"""Class for TFRecord Reader of tf.train.Example"""

from typing import List, Union

import tensorflow as tf

from deepr.readers import base
from deepr.io.path import Path


class TFRecordReader(base.Reader):
    """Class for TFRecord Reader of tf.train.Example.

    Attributes
    ----------
    num_parallel_calls : TYPE
        Description
    num_parallel_reads : int
        Number of parallel reads
    path : List[Union[str, Path]]
        List of filenames or path to directory
    shuffle : bool
        Shuffle files if True before reading.
    """

    def __init__(
        self,
        path: List[Union[str, Path]],
        num_parallel_reads: int = 8,
        num_parallel_calls: int = 8,
        shuffle: bool = True,
    ):
        super().__init__()
        self.path = path
        self.num_parallel_reads = num_parallel_reads
        self.num_parallel_calls = num_parallel_calls
        self.shuffle = shuffle

    def __repr__(self) -> str:
        return f"TFRecordReader(path={self.path})"

    @property
    def filenames(self):
        if isinstance(self.path, list):
            return sorted([str(path) for path in self.path])
        else:
            if Path(self.path).is_dir():
                paths = Path(self.path).glob("*")
                return sorted([str(path) for path in paths if path.is_file() and not path.name.startswith("_")])
            else:
                return [str(self.path)]

    @property
    def compression_type(self):
        if str(self.filenames[0]).endswith("gz"):
            return "GZIP"
        elif str(self.filenames[0]).endswith("zlib"):
            return "ZLIB"
        else:
            return ""

    def as_dataset(self) -> tf.data.Dataset:
        """Build a tf.data.Dataset"""
        if self.shuffle:
            filenames = self.filenames
            dataset = tf.data.Dataset.from_tensor_slices(filenames).shuffle(len(filenames))
            dataset = dataset.interleave(
                lambda filename: tf.data.TFRecordDataset(filename, compression_type=self.compression_type),
                cycle_length=self.num_parallel_reads,
                num_parallel_calls=self.num_parallel_calls,
            )
        else:
            dataset = tf.data.TFRecordDataset(
                self.filenames, compression_type=self.compression_type, num_parallel_reads=self.num_parallel_reads
            )
        return dataset


def bytes_feature(value: List[bytes]):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value: List[float]):
    """Returns an float_list from a float"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value: List[int]):
    """Returns an int64_list from an int"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def int64_feature_list(values: List[List[int]]):
    """Returns a FeatureList from a list of int"""
    input_features = [int64_feature(value) for value in values]
    return tf.train.FeatureList(feature=input_features)
