"""Lookup Preprocessing Utilities."""

from typing import Callable

import tensorflow as tf

from deepr.prepros import base


class TableInitializer(base.Prepro):
    """Table Initializer.

    Tensorflow does not allow tables initialization inside a ``map``
    transformation (all tables must be created outside the ``map``).

    To remedy this, follow this example

    First, create a ``table_initializer_fn`` that uses the
    ``tf.AUTO_REUSE`` argument.

    >>> import deepr as dpr
    >>> def table_initializer_fn():
    ...     return dpr.utils.table_from_mapping(
    ...         name="partner_table", mapping={1: 2}, reuse=tf.AUTO_REUSE
    ... )

    Then, define your preprocessing pipeline as follows

    >>> prepro_fn = dpr.prepros.Serial(
    ...     dpr.prepros.TableInitializer(table_initializer_fn),
    ...     dpr.prepros.Map(dpr.layers.Lookup(table_initializer_fn)),
    ... )

    When applying the ``prepro_fn`` on a ``tf.data.Dataset``, it will
    run the ``table_initializer_fn`` at the beginning (outside the
    ``map`` transformation), then apply the ``Lookup`` that uses the
    same ``table_initializer_fn``, but thanks to ``reuse=tf.AUTO_REUSE``
    instead of creating a new table, it will simply reuse the table
    created by the ``TableInitializer``.
    """

    def __init__(self, table_initializer_fn: Callable[[], tf.contrib.lookup.HashTable]):
        self.table_initializer_fn = table_initializer_fn

    def apply(self, dataset: tf.data.Dataset, mode: str = None) -> tf.data.Dataset:
        self.table_initializer_fn()
        return dataset
