"""Lookup Utilities and Layer"""

from typing import Dict, Callable
import logging

import numpy as np
import tensorflow as tf

from deepr.layers import base
from deepr.utils.field import TensorType


LOGGER = logging.getLogger(__name__)


class Lookup(base.Layer):
    """Lookup Layer

    Attributes
    ----------
    table_initializer_fn : Callable[[], tf.contrib.lookup.HashTable]
        Function that creates a table
    """

    def __init__(self, table_initializer_fn: Callable[[], tf.contrib.lookup.HashTable], **kwargs):
        super().__init__(n_in=1, n_out=1, **kwargs)
        self.table_initializer_fn = table_initializer_fn

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        return self.table_initializer_fn().lookup(tensors)


class LookupFromFile(Lookup):
    """Lookup From File Layer

    Creates a table at runtime from a mapping file. The table will map
    each key to its corresponding line index as an tf.int64.

    Attributes
    ----------
    key_dtype : tf.DType
        Keys type
    path : str
        Path to mapping file
    table_name : str
        Name of the HashTable
    """

    def __init__(self, table_name: str, path: str, key_dtype=None, **kwargs):
        super().__init__(lambda: table_from_file(name=table_name, path=path, key_dtype=key_dtype), **kwargs)
        self.table_name = table_name
        self.path = path
        self.key_dtype = key_dtype


class LookupFromMapping(Lookup):
    """Lookup From Mapping Layer

    Attributes
    ----------
    default_value : Any
        Default value for missing keys
    key_dtype : tf.DType
        Keys type
    mapping : Dict[Any, Any]
        Mapping keys -> index
    table_name : str
        Name of the HashTable
    value_dtype : tf.DType
        Values type
    """

    def __init__(self, table_name: str, mapping: Dict, default_value=None, key_dtype=None, value_dtype=None, **kwargs):
        super().__init__(
            lambda: table_from_mapping(
                name=table_name,
                mapping=mapping,
                default_value=default_value,
                key_dtype=key_dtype,
                value_dtype=value_dtype,
            ),
            **kwargs,
        )
        self.table_name = table_name
        self.mapping = mapping
        self.default_value = default_value
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype


def table_from_file(name: str, path: str, key_dtype=None):
    """Create table from file"""
    LOGGER.info(f"Creating table {name} from {path}")
    table = tf.contrib.lookup.index_table_from_file(vocabulary_file=path, name=name, key_dtype=key_dtype)
    return table


def table_from_mapping(name: str, mapping: Dict, default_value=None, key_dtype=None, value_dtype=None):
    """Create table from mapping"""
    LOGGER.info(f"Creating table {name} from mapping.")

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
    return tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(
            keys=keys_np, values=values_np, key_dtype=key_dtype, value_dtype=value_dtype
        ),
        name=name,
        default_value=default_value,
    )
