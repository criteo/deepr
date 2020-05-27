"""Combine Preprocessors"""

import itertools
import logging
from typing import Tuple, List, Generator, Union

import tensorflow as tf

from deepr.prepros.base import Prepro
from deepr.prepros import core
from deepr.utils.datastruct import to_flat_tuple


LOGGER = logging.getLogger(__name__)


class Serial(Prepro):
    """Chain preprocessors to define complex preprocessing pipelines.

    It will apply each preprocessing step one after the other on each
    element. For performance reasons, it fuses :class:`~Map` and :class:`~Filter`
    operations into single `tf.data` calls.

    For an example, see the following snippet::

        import deepr.layers as dprl
        import deepr.prepros as dprp

        def gen():
            yield {"a": [0], "b": [0, 1]}
            yield {"a": [0, 1], "b": [0]}
            yield {"a": [0, 1], "b": [0, 1]}

        prepro_fn = dprp.Serial(
            dprp.Map(dprl.Sum(inputs=("a", "b"), outputs="c")),
            dprp.Filter(dprl.IsMinSize(inputs="a", outputs="a_size", size=2)),
            dprp.Filter(dprl.IsMinSize(inputs="b", outputs="b_size", size=2)),
        )

        dataset = tf.data.Dataset.from_generator(gen, {"a": tf.int32, "b": tf.int32}, {"a": (None,), "b": (None,)})
        reader = dpr.readers.from_dataset(prepro_fn(dataset))
        expected = [{"a": [0, 1], "b": [0, 1], "c": [0, 2]}]


    Attributes
    ----------
    fuse : bool, Optional
        If True (default), will fuse :class:`~Map` and :class:`~Filter`.
    preprocessors : Union[Prepro, Tuple[Prepro], List[Prepro], Generator[Prepro, None, None]]
        Positional arguments of :class:`~Prepro` instance or Tuple / List /
        Generator of prepro instances
    """

    def __init__(
        self,
        *preprocessors: Union[Prepro, Tuple[Prepro], List[Prepro], Generator[Prepro, None, None]],
        fuse: bool = True,
        num_parallel_calls: int = None,
    ):
        super().__init__()
        self.preprocessors = to_flat_tuple(preprocessors)
        self.fuse = fuse
        self.num_parallel_calls = num_parallel_calls

        # Iterable of preprocessors used by the apply method
        self._preprocessors = (
            _fuse(*self.preprocessors, num_parallel_calls=num_parallel_calls) if fuse else self.preprocessors
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.preprocessors})"

    def apply(self, dataset: tf.data.Dataset, mode: str = None) -> tf.data.Dataset:
        """Pre-process a dataset"""
        for prepro in self._preprocessors:
            dataset = prepro.apply(dataset, mode=mode)
        return dataset


def _fuse(*preprocessors: Prepro, num_parallel_calls: int = None) -> Tuple[Prepro, ...]:
    """Group Map and Filter in _FusedMap and _FusedFilter"""

    def _flatten(prepros):
        for prepro in prepros:
            if isinstance(prepro, Serial):
                yield from _flatten(prepro.preprocessors)
            else:
                yield prepro

    def _prepro_type(prepro: Prepro) -> str:
        if isinstance(prepro, core.Map):
            return "map"
        elif isinstance(prepro, core.Filter):
            return "filter"
        else:
            return "other"

    def _gen():
        for prepro_type, prepros in itertools.groupby(_flatten(preprocessors), _prepro_type):
            if prepro_type == "map":
                yield _FusedMap(*list(prepros), num_parallel_calls=num_parallel_calls)
            elif prepro_type == "filter":
                yield _FusedFilter(*list(prepros))
            else:
                yield list(prepros)

    return to_flat_tuple(_gen())


class _FusedMap(Prepro):
    """Fused Map"""

    def __init__(self, *preprocessors: core.Map, num_parallel_calls: int = None):
        self.preprocessors = preprocessors
        self.num_parallel_calls = num_parallel_calls
        if not all(isinstance(prepro, core.Map) for prepro in self.preprocessors):
            msg = f"All processors must be `deepr.Map` but got {self.preprocessors}"
            raise TypeError(msg)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.preprocessors})"

    def apply(self, dataset: tf.data.Dataset, mode: str = None):
        """Apply preprocessors as one map operation"""
        # Filter preprocessors for this mode
        active_prepros: List[core.Map] = []
        for prepro in self.preprocessors:
            if mode is not None and prepro.modes is not None and mode not in prepro.modes:
                LOGGER.info(f"Not applying {prepro} (mode={mode})")
                continue
            active_prepros.append(prepro)

        # Apply filtered preprocessors
        if not active_prepros:
            return dataset
        else:

            def _fused_tf_map_func(element):
                for prepro in active_prepros:
                    element = prepro.tf_map_func(element)
                return element

            return dataset.map(_fused_tf_map_func, num_parallel_calls=self.num_parallel_calls)


class _FusedFilter(Prepro):
    """Fused Filter"""

    def __init__(self, *preprocessors: core.Filter):
        self.preprocessors = preprocessors
        if not all(isinstance(prepro, core.Filter) for prepro in self.preprocessors):
            msg = f"All processors must be `deepr.Filter` but got {self.preprocessors}"
            raise TypeError(msg)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.preprocessors})"

    def apply(self, dataset: tf.data.Dataset, mode: str = None):
        """Apply preprocessors as one filter operation"""
        # Filter preprocessors for this mode
        active_prepros: List[core.Filter] = []
        for prepro in self.preprocessors:
            if mode is not None and prepro.modes is not None and mode not in prepro.modes:
                LOGGER.info(f"Not applying {prepro} (mode={mode})")
                continue
            active_prepros.append(prepro)

        # Apply filtered preprocessors
        if not active_prepros:
            return dataset
        else:

            def _fused_tf_predicate(element):
                pred = None
                for prepro in active_prepros:
                    new_pred = prepro.tf_predicate(element)
                    pred = new_pred if pred is None else tf.logical_and(pred, new_pred)
                return pred

            return dataset.filter(_fused_tf_predicate)
