"""Upload Configs to MLFlow"""

from dataclasses import dataclass
from typing import Dict, Optional, Callable, Tuple, Any
import logging

from deepr.config.base import fill_macros, TYPE
from deepr.jobs import base
from deepr.utils import mlflow


LOGGER = logging.getLogger(__name__)


@dataclass
class MLFlowSaveConfigs(base.Job):
    """Upload Configs to MLFlow"""

    use_mlflow: Optional[bool] = False
    config: Optional[Dict] = None
    macros: Optional[Dict] = None
    macros_eval: Optional[Dict] = None
    formatter: Optional[Callable[[Dict], Dict]] = None

    def run(self):
        config = self.config if self.config else dict()
        macros = self.macros if self.macros else dict()
        macros_eval = self.macros_eval if self.macros_eval else dict()

        if self.use_mlflow:
            # Prepare configs to upload
            macros_static = {macro: params for macro, params in macros.items() if TYPE not in params}
            macros_no_static = {macro: params for macro, params in macros.items() if TYPE in params}
            config_no_static = fill_macros(config, macros_static)
            config_no_macros = fill_macros(config, macros_eval)

            # Save configs and macros
            mlflow.log_dict(macros, "macros.json")
            mlflow.log_dict(macros_eval, "macros_eval.json")
            mlflow.log_dict(macros_static, "macros_static.json")
            mlflow.log_dict(macros_no_static, "macros_no_static.json")
            mlflow.log_dict(config, "config.json")
            mlflow.log_dict(config_no_static, "config_no_static.json")
            mlflow.log_dict(config_no_macros, "config_no_macros.json")

            # Log config and macros to MLFlow
            formatter = self.formatter if self.formatter else MLFlowFormatter()
            parameters = list(formatter({**macros_eval, **config_no_macros}).items())
            for idx in range(0, len(parameters), 100):
                mlflow.log_params(dict(parameters[idx : idx + 100]))


class MLFlowFormatter:
    """Flattens dictionaries and extract sub-keys

    Example
    -------
    >>> from deepr.jobs import MLFlowFormatter
    >>> params = {
    ...     "foo": {
    ...         "type": "foo.Foo",
    ...         "bar": {
    ...             "x": 1,
    ...             "y": 2,
    ...         }
    ...     }
    ... }
    >>> formatter = MLFlowFormatter(include_keys=("bar", "x"), skip_values=(2,))
    >>> formatter(params)
    {'bar.x': 1, 'x': 1}

    Attributes
    ----------
    include_keys : Tuple[str, ...], Optional
        If not None, keep only dictionaries nested under such keys
    skip_keys : Tuple[str, ...]
        Do not include dictionaries nested under such keys
    skip_values : Tuple[str, ...]
        Do not include such values
    """

    def __init__(
        self,
        include_keys: Optional[Tuple[str, ...]] = None,
        skip_keys: Tuple[str, ...] = (),
        skip_values: Tuple[str, ...] = (),
    ):
        self.include_keys = include_keys
        self.skip_keys = skip_keys
        self.skip_values = skip_values

    def __call__(self, params: Dict) -> Dict:
        """Flatten nested config dictionaries for MLFlow logging."""

        def _format_type(val):
            return val.split(".")[-1]

        def _flatten(data: Dict) -> Dict[str, Any]:
            """Flattens nested dictionary, convert lists into dicts.

            A nested dict is flattened using a "." to join the keys of
            different levels. Values for keys ending with KEY_TYPE are
            shortened (don't keep the full import string but just the
            class's name).
            List of dictionaries are converted to a dictionary whose
            top-level keys are the value associated with KEY_TYPE in
            each of the children.

            Example
            -------
            >>> data = {
            ...     "foo": {
            ...         "type": "foo.Foo",
            ...         "bar": [{"type": "foo.Bar", "baz": 1}]
            ...     }
            ... }  # doctest: +SKIP
            >>> _flatten(data)  # doctest: +SKIP
            {"foo.type": "Foo", "foo.bar.Bar.baz": 1}
            """
            flat = dict()  # type: ignore
            for key, value in data.items():
                if isinstance(value, dict):
                    flat.update({f"{key}.{subkey}": subval for subkey, subval in _flatten(value).items()})
                elif isinstance(value, (list, tuple)) and all(isinstance(val, dict) and TYPE in val for val in value):
                    for item in value:
                        dtype = _format_type(item[TYPE])
                        params = {key: val for key, val in item.items() if key != TYPE}
                        flat.update(_flatten({key: {dtype: params}}))
                elif key == TYPE:
                    flat[key] = _format_type(value)
                else:
                    flat[str(key)] = value
            return flat

        # Flatten dictionary, extract relevant keys
        params = _flatten(params)
        formatted = dict()  # type: Dict[str, Any]
        for param, value in params.items():
            # Slice param to keep only keys after relevant keys
            keys = param.split(".")
            if self.include_keys is not None:
                sliced = []  # type: ignore
                for key in keys:
                    if sliced or key in self.include_keys:
                        sliced.append(key)
                keys = sliced
            if not keys or set(keys) & set(self.skip_keys) or any(str(val) in str(value) for val in self.skip_values):
                continue

            # Update formatted with the sliced param key if not present
            key = ".".join(keys)
            formatted[key] = formatted.get(key, value)

            # Also add the last key if in include_keys if not present
            last = keys[-1]
            if self.include_keys is not None and last in self.include_keys:
                formatted[last] = formatted.get(last, value)

        # Make sure keys are within the 250 characters limit
        filtered = {}
        for key, value in formatted.items():
            if len(str(key)) >= 250:
                LOGGER.error(f"Key too long {key}")
                continue
            if len(str(value)) >= 250:
                LOGGER.error(f"Value too long {value}")
                continue
            filtered[key] = value

        return filtered
