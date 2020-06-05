"""TFRecords writer."""

import logging

import tensorflow as tf

from deepr.io.path import Path
from deepr.readers.base import from_dataset
from deepr.writers import base
from deepr.utils.iter import progress, chunks


LOGGER = logging.getLogger(__name__)


class TFRecordWriter(base.Writer):
    """TFRecords writer."""

    def __init__(self, path: str, chunk_size: int = None, compression_type: str = "GZIP", secs: int = 60):
        self.path = path
        self.chunk_size = chunk_size
        self.compression_type = compression_type
        self.secs = secs

    def write(self, dataset: tf.data.Dataset):
        def _write_chunk(data, path_data):
            if self.compression_type == "GZIP" and not path_data.endswith(".gz"):
                path_data += ".gz"
            elif self.compression_type == "ZLIB" and not path_data.endswith(".zlib"):
                path_data += ".zlib"
            LOGGER.info(f"Writing tf record dataset to {path_data}")
            with tf.io.TFRecordWriter(
                path_data, options=tf.io.TFRecordOptions(compression_type=self.compression_type)
            ) as writer:
                for ex in data:
                    writer.write(ex)

        if self.chunk_size is None:
            _write_chunk(progress(from_dataset(dataset), secs=self.secs), self.path)
        else:
            if not Path(self.path).is_hdfs:
                Path(self.path).mkdir(parents=True, exist_ok=True)
            for idx, chunk in enumerate(chunks(progress(from_dataset(dataset), secs=self.secs), self.chunk_size)):
                path_chunk = f"{self.path}/part-{idx}.tfrecord"
                _write_chunk(chunk, path_chunk)
