"""Core Classes for preprocessing"""

from typing import Callable, Iterable
import logging

import tensorflow as tf

from deepr.utils.field import Field
from deepr.prepros import base
from deepr.layers import Layer


LOGGER = logging.getLogger(__name__)


class Map(base.Prepro):
    """Map a function on each element of a tf.data.Dataset.

    A :class:`~Map` instance applies a ``map_func`` to all elements of a
    dataset. By default, elements are expected to be dictionaries. You
    can set ``on_dict=False`` if your dataset does not yield
    dictionaries.

    If elements are dictionaries, you can use the additional argument
    ``update`` to choose to update dictionaries instead of overriding
    them.

    NOTE: If ``map_func`` is a :class:`~deepr.layers.Layer`, it directly uses ``forward``
    or ``forward_as_dict`` to avoid inspection overhead from the
    ``Layer.__call__`` method.

    WARNING: if ``map_func`` is a ``Layer``, the ``mode`` will not be
    forwarded by the ``Map.apply()`` call, and the default ``None`` will
    always be used. This is intended to keep the signature of the
    generic ``map_func`` in line with the ``tf.Dataset.map`` method.

    If you wish to use a :class:`~deepr.layers.Layer` with a given ``mode``, you can do

    >>> from functools import partial
    >>> from deepr import readers
    >>> from deepr.layers import Sum
    >>> from deepr.prepros import Map
    >>> layer = Sum()
    >>> prepro_fn = Map(partial(layer.forward_as_dict, mode=tf.estimator.ModeKeys.TRAIN))

    For example, by setting `update=True` (DEFAULT behavior)

    >>> def gen():
    ...     yield {"a": 0}
    >>> dataset = tf.data.Dataset.from_generator(gen, {"a": tf.int32}, {"a": tf.TensorShape([])})
    >>> list(readers.from_dataset(dataset))
    [{'a': 0}]
    >>> def map_func(x):
    ...     return {"b": x["a"] + 1}
    >>> prepro_fn = Map(map_func, update=True)
    >>> list(readers.from_dataset(prepro_fn(dataset)))
    [{'a': 0, 'b': 1}]

    On the other hand, ``update=False`` yields the output of the
    ``map_func``

    >>> prepro_fn = Map(map_func, update=False)
    >>> list(readers.from_dataset(prepro_fn(dataset)))
    [{'b': 1}]

    Because some preprocessing pipelines behave differently depending
    on the mode (TRAIN, EVAL, PREDICT), an optional argument can be
    provided. By setting modes, you select the modes on which the map
    transformation should apply. For example:

    >>> prepro_fn = Map(map_func, modes=[tf.estimator.ModeKeys.TRAIN])
    >>> list(readers.from_dataset(prepro_fn(dataset, tf.estimator.ModeKeys.TRAIN)))
    [{'a': 0, 'b': 1}]
    >>> list(readers.from_dataset(prepro_fn(dataset, tf.estimator.ModeKeys.PREDICT)))
    [{'a': 0}]

    If the mode is not given at runtime, the preprocessing is applied.

    >>> list(readers.from_dataset(prepro_fn(dataset)))
    [{'a': 0, 'b': 1}]


    Attributes
    ----------
    map_func : Callable[[Any], Any]
        Function to map to each element
    modes : Iterable[str], Optional
        Active modes for the map (will skip modes not in modes).
        Default is None (all modes are considered active modes).
    num_parallel_calls : int
        Number of threads.
    on_dict : bool
        If True (default), assumes dataset yields dictionaries
    update : bool
        If True (default), combine element and map_func(element)
    """

    def __init__(
        self,
        map_func: Callable,
        on_dict: bool = True,
        update: bool = True,
        modes: Iterable[str] = None,
        num_parallel_calls: int = None,
    ):
        super().__init__()
        self.map_func = map_func
        self.on_dict = on_dict
        self.update = update
        self.modes = modes
        self.num_parallel_calls = num_parallel_calls

        if self.update and not self.on_dict:
            raise ValueError("update=True but on_dict=False (incoherent)")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.map_func})"

    @property
    def tf_map_func(self):
        """Return final map function."""
        map_func = self.map_func
        if isinstance(map_func, Layer):
            map_func = map_func.forward_as_dict if self.on_dict else map_func.forward
        if self.update:
            return lambda x: {**x, **map_func(x)}
        return map_func

    def apply(self, dataset: tf.data.Dataset, mode: str = None):
        if mode is not None and self.modes is not None and mode not in self.modes:
            LOGGER.info(f"Not applying {self} (mode={mode})")
            return dataset
        return dataset.map(self.tf_map_func, num_parallel_calls=self.num_parallel_calls)


class Filter(base.Prepro):
    """Filter a dataset keeping only elements on which predicate is True

    A :class:`~Filter` instance applies a ``predicate`` to all elements of a
    dataset and keeps only element for which predicate returns True.

    By default, elements are expected to be dictionaries. You can set
    ``on_dict=False`` if your dataset does not yield dictionaries.

    Because some preprocessing pipelines behave differently depending
    on the mode (TRAIN, EVAL, PREDICT), an optional argument can be
    provided. By setting modes, you select the modes on which the map
    transformation should apply. For example:

    >>> from deepr import readers
    >>> from deepr.prepros import Filter
    >>> def gen():
    ...     yield {"a": 0}
    ...     yield {"a": 1}
    >>> raw_dataset = tf.data.Dataset.from_generator(gen, {"a": tf.int32}, {"a": tf.TensorShape([])})
    >>> list(readers.from_dataset(raw_dataset))
    [{'a': 0}, {'a': 1}]
    >>> def predicate(x):
    ...     return {"b": tf.equal(x["a"], 0)}
    >>> prepro_fn = Filter(predicate, modes=[tf.estimator.ModeKeys.TRAIN])
    >>> raw_dataset = tf.data.Dataset.from_generator(gen, {"a": tf.int32}, {"a": tf.TensorShape([])})
    >>> dataset = prepro_fn(raw_dataset, tf.estimator.ModeKeys.TRAIN)
    >>> list(readers.from_dataset(dataset))
    [{'a': 0}]

    >>> dataset = prepro_fn(raw_dataset, tf.estimator.ModeKeys.PREDICT)
    >>> list(readers.from_dataset(dataset))
    [{'a': 0}, {'a': 1}]

    If the mode is not given at runtime, the preprocessing is applied.

    >>> dataset = prepro_fn(raw_dataset)
    >>> list(readers.from_dataset(dataset))
    [{'a': 0}]

    Attributes
    ----------
    predicate : Callable
        Predicate function, returns either a tf.bool or a dictionary
        with one key.
    on_dict : bool, Optional
        If True (default), assumes dataset yields dictionaries
    modes : Iterable[str], Optional
        Active modes for the map (will skip modes not in modes).
        Default is None (all modes are considered active modes).
    """

    def __init__(self, predicate: Callable, on_dict: bool = True, modes: Iterable[str] = None):
        super().__init__()
        self.predicate = predicate
        self.on_dict = on_dict
        self.modes = modes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.predicate})"

    @property
    def tf_predicate(self):
        """Return final predicate function."""
        predicate = self.predicate
        if isinstance(predicate, Layer):
            if predicate.n_out != 1:
                msg = f"{predicate} has n_out = {predicate.n_out} (unable to retrieve predicate from layer outputs)"
                raise ValueError(msg)
            return lambda x: predicate.forward_as_dict(x)[predicate.outputs] if self.on_dict else predicate.forward
        if self.on_dict:
            return lambda x: list(predicate(x).values())[0]
        return predicate

    def apply(self, dataset: tf.data.Dataset, mode: str = None):
        if mode is not None and self.modes is not None and mode not in self.modes:
            LOGGER.info(f"Not applying {self} (mode={mode})")
            return dataset
        return dataset.filter(self.tf_predicate)


class Shuffle(base.Prepro):
    """Randomly shuffles the elements of a dataset.

    Attributes
    ----------
    buffer_size : int
        Buffer size for the shuffle buffer
    modes : Iterable[str], Optional
        Active modes for the map (will skip modes not in modes).
        Default is None (all modes are considered active modes).
    """

    def __init__(
        self,
        buffer_size: int,
        modes: Iterable[str] = None,
        seed: tf.int64 = None,
        reshuffle_each_iteration: bool = None,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.modes = modes
        self.seed = seed
        self.reshuffle_each_iteration = reshuffle_each_iteration

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.buffer_size})"

    def apply(self, dataset: tf.data.Dataset, mode: str = None):
        if mode is not None and self.modes is not None and mode not in self.modes:
            LOGGER.info(f"Not applying {self} (mode={mode})")
            return dataset
        return dataset.shuffle(self.buffer_size, seed=self.seed, reshuffle_each_iteration=self.reshuffle_each_iteration)


class Repeat(base.Prepro):
    """Repeats a dataset so each original value is seen count times.

    Attributes
    ----------
    count : int
        Number of dataset repeat, if None or -1, repeat forever.
    modes : Iterable[str], Optional
        Active modes for the map (will skip modes not in modes).
        Default is None (all modes are considered active modes).
    """

    def __init__(self, count: int = None, modes: Iterable[str] = None):
        super().__init__()
        self.count = count
        self.modes = modes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.count})"

    def apply(self, dataset: tf.data.Dataset, mode: str = None):
        if mode is not None and self.modes is not None and mode not in self.modes:
            LOGGER.info(f"Not applying {self} (mode={mode})")
            return dataset
        return dataset.repeat(self.count)


class PaddedBatch(base.Prepro):
    """Combines consecutive elements of a dataset into padded batches.

    NOTE: this applies on dataset yielding dictionaries ONLY.

    If you want to create padded batches from other structures, you
    need to create your own padded batch prepro wrapping the tensorflow
    implementation. For example::

        @dpr.prepros.prepro
        def PaddedBatchDefault(dataset, batch_size, padded_shapes, padding_values):
            return dataset.padded_batch(bath_size, padded_shapes, padding_values)

    Attributes
    ----------
    batch_size : int
        Size of batches
    fields : Iterable[Field]
        Field information for each key of yielded dictionaries
    modes : Iterable[str], Optional
        Active modes for the map (will skip modes not in modes).
        Default is None (all modes are considered active modes).
    """

    def __init__(self, batch_size: int, fields: Iterable[Field], drop_remainder: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.fields = fields
        self.drop_remainder = drop_remainder

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.batch_size})"

    def apply(self, dataset: tf.data.Dataset, mode: str = None):
        # pylint: disable=unused-argument
        padded_shapes = {field.name: field.shape for field in self.fields}
        padding_values = {field.name: tf.constant(field.default, field.dtype) for field in self.fields}
        return dataset.padded_batch(
            self.batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=self.drop_remainder,
        )


class Batch(base.Prepro):
    """Combines consecutive elements of a dataset into batches.

    Attributes
    ----------
    count : int
        Number of dataset repeat
    modes : Iterable[str], Optional
        Active modes for the map (will skip modes not in modes).
        Default is None (all modes are considered active modes).
    """

    def __init__(self, batch_size: int, drop_remainder: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.batch_size})"

    def apply(self, dataset: tf.data.Dataset, mode: str = None):
        # pylint: disable=unused-argument
        return dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)


class Prefetch(base.Prepro):
    """Creates a dataset that prefetch element on CPU / GPU.

    Attributes
    ----------
    buffer_size : int
        Number of element to prefetch.
        High values may lead to high memory consumption, it is
        recommended to use a buffer_size of 1.
    """

    def __init__(self, buffer_size: int):
        super().__init__()
        self.buffer_size = buffer_size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.buffer_size})"

    def apply(self, dataset: tf.data.Dataset, mode: str = None):
        # pylint: disable=unused-argument
        return dataset.prefetch(self.buffer_size)


class Take(base.Prepro):
    """Creates a dataset with at most count elements.

    Attributes
    ----------
    count : int
        Cap the number of elements of a dataset to this number. Using
        None means no capping (will not apply the take transformation).
    """

    def __init__(self, count: int = None):
        super().__init__()
        self.count = count

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.count})"

    def apply(self, dataset: tf.data.Dataset, mode: str = None):
        # pylint: disable=unused-argument
        if self.count is None:
            return dataset
        else:
            return dataset.take(self.count)


class Cache(base.Prepro):
    """Cache Dataset in memory, unless a file is provided.

    You must iterate over the dataset completely to cache it (i.e. a
    ``tf.error.OutOfRangeError`` must be raised).

    If caching to file, note that it consumes a lot of disk space (10x
    to 100x compared to tfrecords), and reloading seems brittle.

    Prefer writing preprocessed data to tfrecord instead.

    Attributes
    ----------
    filename: str
    """

    def __init__(self, filename: str = None, modes: Iterable[str] = None):
        super().__init__()
        self.filename = filename
        self.modes = modes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.filename})"

    def apply(self, dataset: tf.data.Dataset, mode: str = None):
        # pylint: disable=unused-argument
        if mode is not None and self.modes is not None and mode not in self.modes:
            LOGGER.info(f"Not applying {self} (mode={mode})")
            return dataset
        if self.filename:
            return dataset.cache(self.filename)
        else:
            return dataset.cache()
