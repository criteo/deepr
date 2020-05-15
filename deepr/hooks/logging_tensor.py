"""MLFlow Metrics Hook"""

from typing import Any, Dict, List, Callable
import logging
import psutil

import tensorflow as tf
import graphyte

from deepr.hooks.base import TensorHookFactory
from deepr.utils import mlflow


LOGGER = logging.getLogger(__name__)


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
    prefix: str, Optional
        Prefix of tags when sending to MLFlow / Graphite
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

    def __init__(self, tensors: List[str] = None, **kwargs):
        self.tensors = tensors
        self._kwargs = kwargs

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> tf.estimator.LoggingTensorHook:
        if self.tensors is None:
            tensors = {key: tensor for key, tensor in tensors.items() if len(tensor.shape) == 0}
        else:
            tensors = {name: tensors[name] for name in self.tensors}
        return LoggingTensorHook(tensors=tensors, **self._kwargs)


def _default_formatter(tag, value):
    try:
        value = float(value)
        if value == int(value):
            value = int(value)
    except ValueError:
        pass
    return f"{tag} = {value:.7f}" if isinstance(value, float) else f"{tag} = {value}"


class LoggingTensorHook(tf.train.LoggingTensorHook):
    """Logging Hook (tensors and custom metrics as functions)"""

    def __init__(
        self,
        tensors: Dict[str, tf.Tensor],
        functions: Dict[str, Callable[[], float]] = None,
        prefix: str = "",
        use_mlflow: bool = False,
        use_graphite: bool = False,
        config_graphite: Dict = None,
        skip_after_step: int = None,
        formatter: Callable[[str, Any], str] = _default_formatter,
        **kwargs,
    ):
        if "global_step" not in tensors:
            tensors["global_step"] = tf.train.get_global_step()
        super().__init__(tensors, **kwargs)
        self.functions = functions
        self.prefix = prefix
        self.use_mlflow = use_mlflow
        self.use_graphite = use_graphite
        self.config_graphite = config_graphite
        self.skip_after_step = skip_after_step
        self.formatter = formatter

        self._graphite_sender = graphyte.Sender(**config_graphite) if use_graphite else None
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
            if self.use_mlflow:
                mlflow.log_metric(key=f"{self.prefix}{tag}", value=value, step=global_step)
            if self.use_graphite:
                self._graphite_sender.send(tag, value)


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max = None

    def __call__(self):
        memory = super().__call__()
        self._max = memory if self._max is None else max(self._max, memory)
        return self._max
