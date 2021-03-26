"""Convert deepr configs to fromconfig format."""

from collections import Mapping
from pathlib import Path
from typing import List

import fromconfig


class DeeprParser(fromconfig.parser.Parser):
    """Deepr Parser."""

    def __init__(self):
        self.legacy_parser = LegacyParser()
        self.macro_parser = MacroParser()
        self.default_parser = fromconfig.parser.DefaultParser()

    def __call__(self, config: Mapping):
        # Remove special references
        def _remove(item):
            if isinstance(item, str) and item in {"@self", "@macros", "@macros_eval"}:
                return item.replace("@", "_")
            return item

        standard = fromconfig.utils.depth_map(_remove, config)

        # Convert config syntax to new syntax
        converted = self.legacy_parser(standard)

        # Evaluate Macros
        macros_eval = self.macro_parser(converted.get("macros"))

        # Evaluate Config
        parsed = self.default_parser({"config": converted.get("config"), **(macros_eval or {})})

        # Replace special references
        def _references(item):
            if item == "_self":
                return {"_attr_": "fromconfig.Config", "_config_": config.get("config")}
            if item == "_macros":
                return {"_attr_": "fromconfig.Config", "_config_": config.get("macros")}
            if item == "_macros_eval":
                return {"_attr_": "fromconfig.Config", "_config_": macros_eval}
            return item

        return fromconfig.utils.depth_map(_references, parsed.get("config"))


class LegacyParser(fromconfig.parser.Parser):
    """Convert deepr configs to fromconfig format."""

    _RENAME_KEYS = {"type": "_attr_", "*": "_args_", "eval": "_eval_"}

    def __call__(self, config: Mapping):
        def _map_fn(item):
            # Rename keys and handle eval None cases
            if isinstance(item, Mapping):
                result = {}
                for key, value in item.items():
                    key = self._RENAME_KEYS.get(key, key)
                    result[key] = value

                if "_eval_" in result and result["_eval_"] is None:
                    result = {
                        "_attr_": "fromconfig.Config",
                        "_config_": {key: value for key, value in result.items() if key != "_eval_"},
                    }
                return result

            # Replace $param:value by @param.value
            if isinstance(item, str) and item.startswith("$") and not item.startswith("${"):
                parts = item.lstrip("$").split(":")
                return f"@{'.'.join(parts)}"

            return item

        return fromconfig.utils.depth_map(_map_fn, config)


class MacroParser(fromconfig.parser.Parser):
    """Parse and instantiate a macro config."""

    def __call__(self, config: Mapping):
        if config is None:
            return None

        def _to_singleton(item, keys: List[str]):
            if fromconfig.utils.is_mapping(item):
                if "_attr_" in item:
                    return {"_singleton_": ".".join(keys), **item}
                else:
                    return {key: _to_singleton(value, keys + [key]) for key, value in item.items()}
            if fromconfig.utils.is_pure_iterable(item):
                return [_to_singleton(it, keys + [idx]) for idx, it in enumerate(item)]
            return item

        config = _to_singleton(config, [])
        config = fromconfig.parser.SingletonParser()(config)

        def _reference_to_operator(item):
            if fromconfig.utils.is_mapping(item):
                return {key: _reference_to_operator(value) for key, value in item.items()}
            if fromconfig.utils.is_pure_iterable(item):
                return [_reference_to_operator(it) for it in item]
            if isinstance(item, str) and fromconfig.parser.reference.is_reference(item):
                keys = fromconfig.parser.reference.reference_to_keys(item)
                result = config
                for key in keys:
                    if key in result:
                        result = result[key]
                    else:
                        result = {"_attr_": "operator.getitem", "_args_": [result, key]}
                return result
            return item

        config = _reference_to_operator(config)
        config = fromconfig.fromconfig(config)
        config = fromconfig.utils.depth_map(_sanitize, config)
        return config


def _sanitize(item):
    if isinstance(item, Mapping):
        return dict(item)
    if isinstance(item, Path):
        return str(item)
    return item
