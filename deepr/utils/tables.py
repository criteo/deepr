"""Tables Utilities"""

import logging
from typing import Dict

import numpy as np
import tensorflow as tf

from deepr.utils.field import TensorType


LOGGER = logging.getLogger(__name__)


class TableContext:
    """Context Manager to reuse Tensorflow tables.

    Tensorflow does not have a ``tf.get_variable`` equivalent for
    tables. The ``TableContext`` is here to provide this functionality.

    Example
    -------
    >>> import deepr as dpr
    >>> with dpr.utils.TableContext() as tables:
    ...     table = dpr.utils.table_from_mapping(name="my_table", mapping={1: 2})
    ...     tables.get("my_table") is table
    True

    >>> with dpr.utils.TableContext():
    ...     table = dpr.utils.table_from_mapping(name="my_table", mapping={1: 2})
    ...     reused = dpr.utils.table_from_mapping(name="my_table", reuse=True)
    ...     table is reused
    True
    """

    _ACTIVE = None

    def __init__(self):
        if TableContext._ACTIVE is not None:
            msg = "TableContext already active."
            raise ValueError(msg)
        TableContext._ACTIVE = self
        self._tables = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __contains__(self, name: str):
        return name in self._tables

    def close(self):
        TableContext._ACTIVE = None
        self._tables.clear()

    def get(self, name: str):
        if name not in self._tables:
            msg = f"Table '{name}' not in tables. Did you forget a reuse=True?"
            raise KeyError(msg)
        return self._tables[name]

    def set(self, name: str, table):
        if name in self._tables:
            msg = f"Table '{name}' already exists. Did you forget a reuse=True?"
            raise ValueError(msg)
        self._tables[name] = table

    @classmethod
    def is_active(cls):
        return cls._ACTIVE is not None

    @classmethod
    def active(cls):
        if cls._ACTIVE is None:
            msg = f"No active TableContext found. Wrap your code in a `with TableContext():`"
            raise ValueError(msg)
        return cls._ACTIVE


def table_from_file(name: str, path: str = None, key_dtype=None, reuse: bool = False):
    """Create table from file"""
    if reuse is True or (reuse is tf.AUTO_REUSE and name in TableContext.active()):
        return TableContext.active().get(name)
    else:
        LOGGER.info(f"Creating table {name} from {path}")
        if path is None:
            raise ValueError("Path cannot be None")
        table = tf.contrib.lookup.index_table_from_file(vocabulary_file=path, name=name, key_dtype=key_dtype)
        if TableContext.is_active():
            TableContext.active().set(name=name, table=table)
        return table


def index_to_string_table_from_file(
    name: str, path: str = None, vocab_size: int = None, default_value="UNK", reuse: bool = False
):
    """Create reverse table from file"""
    if reuse is True or (reuse is tf.AUTO_REUSE and name in TableContext.active()):
        return TableContext.active().get(name)
    else:
        LOGGER.info(f"Creating reverse table {name} from {path}")
        if path is None:
            raise ValueError("Path cannot be None")
        table = tf.contrib.lookup.index_to_string_table_from_file(
            vocabulary_file=path, name=name, vocab_size=vocab_size, default_value=default_value
        )
        if TableContext.is_active():
            TableContext.active().set(name=name, table=table)
        return table


def table_from_mapping(
    name: str, mapping: Dict = None, default_value=None, key_dtype=None, value_dtype=None, reuse: bool = False
):
    """Create table from mapping"""
    if reuse is True or (reuse is tf.AUTO_REUSE and name in TableContext.active()):
        return TableContext.active().get(name)
    else:
        LOGGER.info(f"Creating table {name} from mapping.")
        if mapping is None:
            raise ValueError("Mapping cannot be None")

        # Convert mapping to arrays of keys and values
        keys, values = zip(*mapping.items())  # type: ignore
        keys_np = np.array(keys)
        values_np = np.array(values)

        # Infer default value if not given
        if default_value is None:
            default_value = TensorType(type(values_np[0].item())).default

        # Infer types if not given
        if key_dtype is None:
            key_dtype = TensorType(type(keys_np[0].item())).tf
        if value_dtype is None:
            value_dtype = TensorType(type(values_np[0].item())).tf

        # Create table
        table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(
                keys=keys_np, values=values_np, key_dtype=key_dtype, value_dtype=value_dtype
            ),
            name=name,
            default_value=default_value,
        )
        if TableContext.is_active():
            TableContext.active().set(name=name, table=table)
        return table
