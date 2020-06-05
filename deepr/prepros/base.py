"""Abstract Base Class for preprocessing"""

from abc import ABC, abstractmethod
import logging
import functools
from typing import Callable, Type, Optional
import inspect

import tensorflow as tf

from deepr.utils.tables import TableContext


LOGGER = logging.getLogger(__name__)


class Prepro(ABC):
    """Base class for composable preprocessing functions.

    `Prepro` are the basic building blocks of a preprocessing pipeline.
    A `Prepro` defines a function on a tf.data.Dataset.

    The basic usage of a :class:`~Prepro` is to apply it on a Dataset. For
    example:
    >>> from deepr import readers
    >>> from deepr.prepros import Map
    >>> def gen():
    ...     for i in range(3):
    ...         yield {"a": i}
    >>> raw_dataset = tf.data.Dataset.from_generator(gen, {"a": tf.int32}, {"a": tf.TensorShape([])})
    >>> list(readers.from_dataset(raw_dataset))
    [{'a': 0}, {'a': 1}, {'a': 2}]
    >>> prepro_fn = Map(lambda x: {'a': x['a'] + 1})
    >>> dataset = prepro_fn(raw_dataset)
    >>> list(readers.from_dataset(dataset))
    [{'a': 1}, {'a': 2}, {'a': 3}]

    Because some preprocessing pipelines behave differently depending
    on the mode (TRAIN, EVAL, PREDICT), an optional argument can be
    provided:
    >>> def map_func(element, mode=None):
    ...     if mode == tf.estimator.ModeKeys.PREDICT:
    ...         return {'a': 0}
    ...     else:
    ...         return element
    >>> prepro_fn = Map(map_func)
    >>> list(readers.from_dataset(raw_dataset))
    [{'a': 0}, {'a': 1}, {'a': 2}]
    >>> dataset = prepro_fn(raw_dataset, mode=tf.estimator.ModeKeys.TRAIN)
    >>> list(readers.from_dataset(dataset))
    [{'a': 0}, {'a': 1}, {'a': 2}]
    >>> dataset = prepro_fn(raw_dataset, mode=tf.estimator.ModeKeys.PREDICT)
    >>> list(readers.from_dataset(dataset))
    [{'a': 0}, {'a': 1}, {'a': 2}]

    TODO: Actually mode in map_func is not taken into account

    :class:`~Map`, :class:`~Filter`, :class:`~Shuffle` and :class:`~Repeat` have a special attribute
    `modes` that you can use to specify the modes on which the
    preprocessing should be applied. For example:
    >>> def map_func(element, mode=None):
    ...     return {'a': 0}
    >>> prepro_fn = Map(map_func, modes=[tf.estimator.ModeKeys.PREDICT])
    >>> dataset = prepro_fn(raw_dataset, tf.estimator.ModeKeys.TRAIN)
    >>> list(readers.from_dataset(dataset))
    [{'a': 0}, {'a': 1}, {'a': 2}]
    >>> dataset = prepro_fn(dataset, tf.estimator.ModeKeys.PREDICT)
    >>> list(readers.from_dataset(dataset))
    [{'a': 0}, {'a': 0}, {'a': 0}]

    Authors of new :class:`~Prepro` subclasses typically override the `apply`
    method of the base :class:`~Prepro` class::

        def apply(self, dataset: tf.data.Dataset, mode: str = None) -> tf.data.Dataset:
            return dataset

    The easiest way to define custom preprocessors is to use the
    `prepro` decorator (see documentation).
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def __call__(self, dataset: tf.data.Dataset, mode: str = None) -> tf.data.Dataset:
        """Alias for apply"""
        if TableContext.is_active():
            return self.apply(dataset, mode=mode)
        else:
            with TableContext():
                return self.apply(dataset, mode=mode)

    @abstractmethod
    def apply(self, dataset: tf.data.Dataset, mode: str = None) -> tf.data.Dataset:
        """Pre-process a dataset"""
        raise NotImplementedError()


class PreproFn(Prepro):
    """Prepro from function."""

    def __init__(self, prepro_fn: Callable[[tf.data.Dataset, Optional[str]], tf.data.Dataset]):
        self.prepro_fn = prepro_fn

    def apply(self, dataset: tf.data.Dataset, mode: str = None) -> tf.data.Dataset:
        return self.prepro_fn(dataset, mode)


def prepro(fn: Callable) -> Type[Prepro]:
    """Decorator that creates a :class:`~Prepro` class from a function.

    For example, the following snippet defines a subclass of
    :class:`~Prepro` whose `apply` offsets each element of the dataset
    by `offset`:

    >>> from deepr import readers
    >>> from deepr.prepros import prepro
    >>> @prepro
    ... def AddOffset(dataset, mode, offset):
    ...     return dataset.map(lambda element: element + offset)
    >>> raw_dataset = tf.data.Dataset.from_tensor_slices([0, 1, 2])
    >>> prepro_fn = AddOffset(offset=1)
    >>> dataset = prepro_fn(raw_dataset)
    >>> list(readers.from_dataset(dataset))
    [1, 2, 3]

    The class created by the decorator is roughly equivalent to

    .. code-block:: python

        class AddOffset(Prepro):

            def __init__(self, offset)
                Prepro.__init__(self)
                self.offset = offset

            def apply(self, dataset, mode: str = None):
                return dataset.map(lambda element: element + self.offset)

    You can also add a 'mode' argument to your preprocessor like so
    >>> @prepro
    ... def AddOffsetInTrain(dataset, mode, offset):
    ...     if mode == tf.estimator.ModeKeys.TRAIN:
    ...         return dataset.map(lambda element: element + offset)
    ...     else:
    ...         return dataset
    >>> prepro_fn = AddOffsetInTrain(offset=1)
    >>> dataset = prepro_fn(raw_dataset, tf.estimator.ModeKeys.TRAIN)
    >>> list(readers.from_dataset(dataset))
    [1, 2, 3]
    >>> dataset = prepro_fn(raw_dataset, tf.estimator.ModeKeys.PREDICT)
    >>> list(readers.from_dataset(dataset))
    [0, 1, 2]
    >>> dataset = prepro_fn(raw_dataset)
    >>> list(readers.from_dataset(dataset))
    [0, 1, 2]

    Note that 'dataset' and 'mode' need to be the the first arguments
    of the function IN THIS ORDER.
    """
    # pylint: disable=protected-access,invalid-name
    parameters = inspect.signature(fn).parameters
    signature = inspect.Signature([param for key, param in parameters.items() if key not in {"dataset", "mode"}])

    # Check parameters
    if list(parameters.keys())[0] != "dataset":
        raise TypeError(f"'dataset' should be the first parameter of {fn.__name__}")
    if "mode" in parameters:
        if list(parameters.keys())[1] != "mode":
            raise TypeError(f"'mode' should be the second parameter of {fn.__name__}")

    @functools.wraps(fn)
    def _init(self, *args, **kwargs):
        Prepro.__init__(self)
        signature.bind(*args, **kwargs)
        self._args = args
        self._kwargs = kwargs

    if "mode" in parameters:

        def _apply(self, dataset, mode: str = None):
            return fn(dataset, mode, *self._args, **self._kwargs)

    else:

        def _apply(self, dataset, mode: str = None):
            # pylint: disable=unused-argument
            return fn(dataset, *self._args, **self._kwargs)

    attributes = {"__module__": fn.__module__, "__doc__": fn.__doc__, "__init__": _init, "apply": _apply}
    return type(fn.__name__, (Prepro,), attributes)
