"""MLFlow Metrics Hook"""

from typing import Any, Dict, List, Callable
import logging
import psutil

import tensorflow as tf

from deepr.hooks.base import TensorHookFactory
from deepr.utils import mlflow
from deepr.utils import graphite
from deepr.metrics import sanitize_metric_name


LOGGER = logging.getLogger(__name__)


def _default_formatter(tag, value):
    try:
        value = float(value)
        if value == int(value):
            value = int(value)
    except ValueError:
        pass
    return f"{tag} = {value:.7f}" if isinstance(value, float) else f"{tag} = {value}"


class LoggingTensorHookFactory(TensorHookFactory):
    """Parametrize the creation of a LoggingTensorHook factory.

    Arguments for instantiation should be provided as keyword arguments.

    Attributes
    ----------
    tensors : List[str], Optional
        Name of the tensors to use at runtime. If None (default), log
        all scalars.
    functions : Dict[str, Callable[[], float]], Optional
        Additional "python" metrics. Each function should return a float
    name: str, Optional
        Name used as prefix of tags when sending to MLFlow / Graphite
    use_mlflow: bool, Optional
        If True, send metrics to MLFlow. Default is False.
    use_graphite: bool, Optional
        If True, send metrics to Graphite. Default is False.
    skip_after_step: int, Optional
        If not None, do not run the hooks after this step.

        Prevents outliers when used in conjunction with an early
        stopping hook that overrides the global_step.
    formatter: Callable[[str, Any], str], Optional
        Formatter for logging, default uses 7 digits precision.
    """

    def __init__(
        self,
        tensors: List[str] = None,
        functions: Dict[str, Callable[[], float]] = None,
        name: str = None,
        use_mlflow: bool = False,
        use_graphite: bool = False,
        skip_after_step: int = None,
        every_n_iter: int = None,
        every_n_secs: int = None,
        at_end: bool = False,
        formatter: Callable[[str, Any], str] = _default_formatter,
    ):
        self.tensors = tensors
        self.functions = functions
        self.name = name
        self.use_mlflow = use_mlflow
        self.use_graphite = use_graphite
        self.skip_after_step = skip_after_step
        self.every_n_iter = every_n_iter
        self.every_n_secs = every_n_secs
        self.at_end = at_end
        self.formatter = formatter

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> tf.estimator.LoggingTensorHook:
        if self.tensors is None:
            tensors = {key: tensor for key, tensor in tensors.items() if len(tensor.shape) == 0}
        else:
            tensors = {name: tensors[name] for name in self.tensors}
        return LoggingTensorHook(
            tensors=tensors,
            functions=self.functions,
            name=self.name,
            use_mlflow=self.use_mlflow,
            use_graphite=self.use_graphite,
            skip_after_step=self.skip_after_step,
            every_n_iter=self.every_n_iter,
            every_n_secs=self.every_n_secs,
            at_end=self.at_end,
            formatter=self.formatter,
        )


class LoggingTensorHook(tf.train.LoggingTensorHook):
    """Logging Hook (tensors and custom metrics as functions)"""

    def __init__(
        self,
        tensors: Dict[str, tf.Tensor],
        functions: Dict[str, Callable[[], float]] = None,
        name: str = None,
        use_mlflow: bool = False,
        use_graphite: bool = False,
        skip_after_step: int = None,
        every_n_iter: int = None,
        every_n_secs: int = None,
        at_end: bool = False,
        formatter: Callable[[str, Any], str] = _default_formatter,
    ):
        if "global_step" not in tensors:
            tensors["global_step"] = tf.train.get_global_step()
        super().__init__(tensors=tensors, every_n_iter=every_n_iter, every_n_secs=every_n_secs, at_end=at_end)
        self.functions = functions
        self.name = name
        self.use_mlflow = use_mlflow
        self.use_graphite = use_graphite
        self.skip_after_step = skip_after_step
        self.formatter = formatter

        self._fn_order = sorted(self.functions.keys()) if self.functions else []

    def _log_tensors(self, tensor_values):
        """Update timer, log tensors, send to MLFlow and Graphite"""
        self._timer.update_last_triggered_step(self._iter_count)
        global_step = tensor_values["global_step"]
        if self.skip_after_step is not None and global_step >= self.skip_after_step:
            return

        # Log tensor and function values
        ord_tensor_values = [(tag, tensor_values[tag]) for tag in self._tag_order]
        ord_function_values = [(tag, self.functions[tag]()) for tag in self._fn_order] if self.functions else []
        LOGGER.info(", ".join(self.formatter(tag, value) for tag, value in ord_tensor_values + ord_function_values))

        # Send to MLFlow and Graphite
        for tag, value in ord_tensor_values + ord_function_values:
            if self.use_graphite:
                graphite.log_metric(tag, value, postfix=self.name)
            if self.use_mlflow:
                tag = tag if self.name is None else f"{self.name}_{tag}"
                mlflow.log_metric(sanitize_metric_name(tag), value, step=global_step)


_UNIT_TO_FACTOR = {"kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3}


class ResidentMemory:
    """Measure resident memory of the current process"""

    def __init__(self, unit: str = "gb"):
        self.unit = unit
        self._factor = _UNIT_TO_FACTOR[self.unit]
        self._process = psutil.Process()

    def __call__(self):
        return self._process.memory_info().rss / self._factor


class MaxResidentMemory(ResidentMemory):
    """Measure maximum resident memory of the current process"""

    def __init__(self, unit: str = "gb"):
        super().__init__(unit=unit)
        self._max = None

    def __call__(self):
        memory = super().__call__()
        self._max = memory if self._max is None else max(self._max, memory)
        return self._max
