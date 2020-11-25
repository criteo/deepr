"""Test for io.parquet"""

import numpy as np
import pandas as pd
import pyarrow as pa

import deepr


def test_io_parquet_dataset_read(tmpdir):
    """Test ParquetDataset"""
    path = str(tmpdir.join("df.parquet.snappy"))
    df = pd.DataFrame(data={"x": [0, 1], "y": [0, 2]})
    df.to_parquet(path)
    with deepr.io.ParquetDataset(path).open() as ds:
        got = ds.read_pandas().to_pandas()
    assert df.equals(got)


def test_io_parquet_dataset_schema(tmpdir):
    """Test schema support"""
    path = str(tmpdir.join("df.parquet.snappy"))
    df = pd.DataFrame(data={"embedding": [np.random.random([5]).astype(np.float32).tolist()]})
    schema = pa.schema([("embedding", pa.list_(pa.float32()))])
    with deepr.io.ParquetDataset(path).open() as ds:
        ds.write_pandas(df, schema=schema)
    with deepr.io.ParquetDataset(path).open() as ds:
        reloaded = ds.read_pandas().to_pandas()
    assert reloaded.iloc[0].embedding.dtype == np.dtype("float32")
