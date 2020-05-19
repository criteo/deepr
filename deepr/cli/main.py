"""Main Entry Point for deepr commands.

Available commands::

    deepr run job.json macros.json

    deepr from_config_and_macros config.json macros.json

    deepr download_config_and_macros_from_mlflow 12dk242jd http://mlflow.url

    deepr add_macro config.json macros.json batch_size,learning_rate

"""

import logging
from typing import List, Union

import fire

from deepr.config.base import parse_config, from_config
from deepr.config.experimental import add_macro_params, find_values
from deepr.config.macros import ismacro
from deepr.jobs.base import Job
from deepr.io.json import load_json, read_json, write_json
from deepr.utils import mlflow


LOGGER = logging.getLogger(__name__)


def run(job: str, macros: str = None):
    """Instantiate job from job configs and macros and run.

    Parameters
    ----------
    job : str
        Path to json file or json string
    macros : str, optional
        Path to json file or json string
    """
    job = from_config_and_macros(job, macros)
    if not isinstance(job, Job):
        raise TypeError(f"Expected Job, but got {type(job)}")
    job.run()


def from_config_and_macros(config: str, macros: str = None):
    """Instantiate object from config and macros.

    Parameters
    ----------
    config : str
        Path to json file or json string
    macros : str, optional
        Path to json file or json string

    Returns
    -------
    Instance
        Defined by config
    """
    parsed = parse_config(load_json(config), load_json(macros) if macros else None)
    return from_config(parsed)


def _from_config(config: str):
    """Instantiate object from parsed config.

    Parameters
    ----------
    config : str
        Path to json file or json string
    """
    return from_config(load_json(config))


def download_config_and_macros_from_mlflow(
    run_id: str,
    tracking_uri: str = None,
    config: str = "config_no_static.json",
    macros: str = "macros_no_static.json",
    path_config: str = "config.json",
    path_macros: str = "macros.json",
):
    """Download config and macros from MLFlow.

    Parameters
    ----------
    run_id : str
        MLFlow Run ID
    tracking_uri : str, optional
        MLFlow tracking URI
    config : str, optional
        Name of the config artifact
    macros : str, optional
        Name of the macros artifact
    path_config : str, optional
        Local path to file where to write the config
    path_macros : str, optional
        Local path to file where to write the macros
    """
    mlflow.download_artifacts(run_id=run_id, path=config, dst_path=path_config, tracking_uri=tracking_uri)
    mlflow.download_artifacts(run_id=run_id, path=macros, dst_path=path_macros, tracking_uri=tracking_uri)


def add_macro(config: str, macros: str, params: Union[List[str], str], macro: str = "params"):
    """Create new params macro, infer new references in config.

    Look for keys in dictionaries of both macros and config that are in
    params, and for each param, store the existing value, replace it by
    a macro reference "$macro:param", and finally add the new macro
    parameter to the macros.

    The resulting updated config and macros are written in the a
    subdirectory of the config's directory, with name "new".

    WARNING: This function is performing a lot of magic by automatically
    replacing values in both macros and config. It is highly recommended
    to manually inspect the resulting config.

    Parameters
    ----------
    config : str
        Path to config.json
    macros : str
        Path to macros.json
    params : Union[List[str], str]
        List of new parameters
    macro : str, optional
        Name of the new macro

    Raises
    ------
    ValueError
        If any param in params has no match in either config and macros.
    """
    # Load config and macros
    params = params.split(",") if isinstance(params, str) else params
    config_dict = read_json(config)
    macros_dict = read_json(macros)
    config_and_macros = {"config": config_dict, "macros": macros_dict}

    # Retrieve existing values, prefer values from macros
    LOGGER.info("Automatically retrieving existing values for new parameters.")
    params_values = {**find_values(config_dict, params), **find_values(macros_dict, params)}

    # Add new macro params in config and macros
    LOGGER.info("Automatically adding new macro params in config and macros.")
    config_and_macros = add_macro_params(config_and_macros, macro=macro, params=params)
    new_config_dict = config_and_macros["config"]
    new_macros_dict = config_and_macros["macros"]

    # Update macros with the new macro
    LOGGER.info(f"Building new macro {macro} with parameters :")
    new_macros_dict[macro] = new_macros_dict.get(macro, {})
    for param in params:
        value = params_values.get(param, f"${macro}:{param}")
        new_macros_dict[macro][param] = value
        if ismacro(value):
            LOGGER.warning(f"- {param}: {value} IS NOT SET (manual fix required if not using `ParamsTuner`).")
        else:
            LOGGER.info(f"- {param}: {value}")

    # Write to new
    write_json(new_config_dict, config)
    write_json(new_macros_dict, macros)


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)
    fire.Fire(
        {
            "run": run,
            "from_config": _from_config,
            "from_config_and_macros": from_config_and_macros,
            "download_config_and_macros_from_mlflow": download_config_and_macros_from_mlflow,
            "add_macro": add_macro,
        }
    )


if __name__ == "__main__":
    main()
