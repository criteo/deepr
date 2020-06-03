"""Lookup Utilities and Layer"""

from typing import Dict, Callable
import logging

import tensorflow as tf

from deepr.layers import base
from deepr.utils.tables import table_from_mapping, table_from_file, index_to_string_table_from_file


LOGGER = logging.getLogger(__name__)


class Lookup(base.Layer):
    """Lookup Layer.

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
    """Lookup From File Layer.

    Creates a table at runtime from a mapping file. The table will map
    each key to its corresponding line index as an tf.int64.

    Attributes
    ----------
    key_dtype : tf.DType
        Keys type
    path : str
        Path to mapping file
    reuse : bool
        If True, reuse table with the same name
    table_name : str
        Name of the HashTable
    """

    def __init__(self, table_name: str, path: str, key_dtype=None, reuse: bool = False, **kwargs):
        super().__init__(
            lambda: table_from_file(name=table_name, path=path, key_dtype=key_dtype, reuse=reuse), **kwargs
        )
        self.table_name = table_name
        self.path = path
        self.key_dtype = key_dtype
        self.reuse = reuse


class LookupFromMapping(Lookup):
    """Lookup From Mapping Layer.

    Attributes
    ----------
    default_value : Any
        Default value for missing keys
    key_dtype : tf.DType
        Keys type
    mapping : Dict[Any, Any]
        Mapping keys -> index
    reuse : bool
        If True, reuse the layer with the same name
    table_name : str
        Name of the HashTable
    value_dtype : tf.DType
        Values type
    """

    def __init__(
        self,
        table_name: str,
        mapping: Dict,
        default_value=None,
        key_dtype=None,
        value_dtype=None,
        reuse: bool = False,
        **kwargs,
    ):
        super().__init__(
            lambda: table_from_mapping(
                name=table_name,
                mapping=mapping,
                default_value=default_value,
                key_dtype=key_dtype,
                value_dtype=value_dtype,
                reuse=reuse,
            ),
            **kwargs,
        )
        self.table_name = table_name
        self.mapping = mapping
        self.default_value = default_value
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        self.reuse = reuse


class LookupIndexToString(Lookup):
    """Lookup Index To String.

    Creates a table at runtime from a mapping file. The table will map
    each key to its corresponding line index as an tf.int64.

    Attributes
    ----------
    default_value : Any
        Default Value for missing keys
    path : str
        Path to mapping file
    reuse : bool
        If True, reuse the table with the same name
    table_name : str
        Name of the HashTable
    vocab_size : int
        Size of the vocab
    """

    def __init__(
        self,
        table_name: str,
        path: str = None,
        vocab_size: int = None,
        default_value="UNK",
        reuse: bool = False,
        **kwargs,
    ):
        super().__init__(
            lambda: index_to_string_table_from_file(
                name=table_name, path=path, vocab_size=vocab_size, default_value=default_value, reuse=reuse
            ),
            **kwargs,
        )
        self.table_name = table_name
        self.path = path
        self.vocab_size = vocab_size
        self.default_value = default_value
        self.reuse = reuse
