"""Utilities for parquet"""

from contextlib import contextmanager
import logging
from typing import Union, List

import pandas as pd
import numpy as np
import pyarrow
import pyarrow.parquet as pq
from pyarrow.filesystem import FileSystem

from deepr.io.path import Path
from deepr.io.hdfs import HDFSFileSystem


LOGGER = logging.getLogger(__name__)


class ParquetDataset:
    """Context aware ParquetDataset with support for chunk writing.

    Makes it easier to read / write :class:`~ParquetDataset`. For example

    >>> from deepr.io import ParquetDataset
    >>> df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    >>> with ParquetDataset("viewfs://root/foo.parquet.snappy").open() as ds:  # doctest: +SKIP
    ...     ds.write_pandas(df, chunk_size=100)  # doctest: +SKIP

    The use of context managers automatically opens / closes the dataset
    as well as the connection to the FileSystem.

    Attributes
    ----------
    path_or_paths : Union[str, Path, List[Union[str, Path]]]
        Path to parquet dataset (directory or file), or list of files.
    filesystem : FileSystem, Optional
        FileSystem, if None, will be inferred automatically later.
    """

    def __init__(
        self,
        path_or_paths: Union[str, Path, List[Union[str, Path]]],
        filesystem: FileSystem = None,
        metadata: pq.FileMetaData = None,
        schema: pyarrow.Schema = None,
        split_row_groups: bool = False,
        validate_schema: bool = True,
        filters: List = None,
        metadata_nthreads: int = 1,
        memory_map: bool = False,
    ):
        self.path_or_paths = list(map(str, path_or_paths)) if isinstance(path_or_paths, list) else str(path_or_paths)
        self.filesystem = filesystem
        self.schema = schema
        self.metadata = metadata
        self.split_row_groups = split_row_groups
        self.validate_schema = validate_schema
        self.filters = filters
        self.metadata_nthreads = metadata_nthreads
        self.memory_map = memory_map

    @property
    def pq_dataset(self):
        return pq.ParquetDataset(
            path_or_paths=self.path_or_paths,
            filesystem=self.filesystem,
            metadata=self.metadata,
            schema=self.schema,
            split_row_groups=self.split_row_groups,
            validate_schema=self.validate_schema,
            filters=self.filters,
            metadata_nthreads=self.metadata_nthreads,
            memory_map=self.memory_map,
        )

    @property
    def is_hdfs(self) -> bool:
        if isinstance(self.path_or_paths, str):
            return Path(self.path_or_paths).is_hdfs  # type: ignore
        else:
            return Path(self.path_or_paths[0]).is_hdfs

    @property
    def is_local(self) -> bool:
        return not self.is_hdfs

    @contextmanager
    def open(self):
        """Open HDFS Filesystem if dataset on HDFS"""
        if self.filesystem is None and self.is_hdfs:
            with HDFSFileSystem() as hdfs:
                yield ParquetDataset(
                    path_or_paths=self.path_or_paths,
                    filesystem=hdfs,
                    metadata=self.metadata,
                    schema=self.schema,
                    split_row_groups=self.split_row_groups,
                    validate_schema=self.validate_schema,
                    filters=self.filters,
                    metadata_nthreads=self.metadata_nthreads,
                    memory_map=self.memory_map,
                )
        else:
            yield self

    def read(self, columns: List[str] = None, use_threads: bool = True, use_pandas_metadata: bool = False):
        return self.pq_dataset.read(columns=columns, use_threads=use_threads, use_pandas_metadata=use_pandas_metadata)

    def read_pandas(self, columns: List[str] = None, use_threads: bool = True):
        return self.pq_dataset.read_pandas(columns=columns, use_threads=use_threads)

    def write(self, table: pyarrow.Table, compression="snappy"):
        if not isinstance(self.path_or_paths, str):
            msg = f"Cannot write table to {self.path_or_paths} (expected string)"
            raise TypeError(msg)
        LOGGER.info(f"Writing table to {self.path_or_paths}")
        with Path(self.path_or_paths).open("wb", filesystem=self.filesystem) as file:
            pq.write_table(table, file, compression=compression)

    def write_pandas(
        self,
        df: pd.DataFrame,
        compression="snappy",
        num_chunks: int = None,
        chunk_size: int = None,
        schema: pyarrow.Schema = None,
    ):
        """Write DataFrame as Parquet Dataset"""
        # Check arguments
        if not isinstance(self.path_or_paths, str):
            msg = f"Cannot write table to {self.path_or_paths} (expected string)"
            raise TypeError(msg)
        if num_chunks is not None and chunk_size is not None:
            msg = "Both num_chunks and chunk_size are given, not allowed"
            raise ValueError(msg)
        if chunk_size is not None:
            num_chunks = max(len(df) // chunk_size, 1)

        # Write DataFrame to parquet
        if num_chunks is None:
            table = pyarrow.Table.from_pandas(df, schema=schema, preserve_index=False)
            self.write(table, compression=compression)
        else:
            Path(self.path_or_paths).mkdir(parents=True, exist_ok=True, filesystem=self.filesystem)
            chunks = np.array_split(df, num_chunks)
            for idx, chunk in enumerate(chunks):
                filename = f"part-{idx:05d}.parquet.{compression}"
                chunk_path = Path(self.path_or_paths, filename)
                LOGGER.info(f"Writing chunk:{idx} to {chunk_path}")
                with chunk_path.open("wb", filesystem=self.filesystem) as file:
                    table = pyarrow.Table.from_pandas(chunk, schema=schema, preserve_index=False)
                    pq.write_table(table, file, compression=compression)
