"""Save Dataset Job."""

import dataclasses
import logging
from typing import Callable, List, Optional

import tensorflow as tf
from deepr.jobs import base
from deepr.utils.field import Field
from deepr.writers.record import TFRecordWriter
from deepr.prepros.base import PreproFn
from deepr.prepros.combinators import Serial
from deepr.prepros.record import ToExample


LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class SaveDataset(base.Job):
    """Save Dataset Job."""

    input_fn: Callable[[], tf.data.Dataset]
    path: str
    fields: List[Field]
    prepro_fn: Callable[[tf.data.Dataset, str], tf.data.Dataset] = dataclasses.field(default=lambda dataset, _: dataset)
    num_parallel_calls: Optional[int] = None
    chunk_size: Optional[int] = None
    compression_type: str = "GZIP"
    secs: Optional[int] = 60
    mode: Optional[str] = None

    def run(self):
        LOGGER.info(f"Saving dataset to {self.path}")
        prepro_fn = Serial(
            PreproFn(self.prepro_fn), ToExample(fields=self.fields), num_parallel_calls=self.num_parallel_calls
        )
        writer = TFRecordWriter(
            path=self.path, chunk_size=self.chunk_size, compression_type=self.compression_type, secs=self.secs
        )
        writer.write(prepro_fn(self.input_fn(), mode=self.mode))
