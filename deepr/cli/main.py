"""Main Entry Point"""

import logging
from shutil import copyfile
from typing import List, Union

import fire
import mlflow

from deepr.jobs.base import Job
from deepr.config.base import parse_config, from_config
from deepr.config.experimental import add_macro_params
from deepr.io.json import read_json, write_json


LOGGER = logging.getLogger(__name__)


def run(job: str, macros: str = None):
    """Instantiate job from config and macros and run"""
    job = from_config_file(job, macros)
    if not isinstance(job, Job):
        raise TypeError(f"Expected job of type {Job} but got {type(job)}")
    job.run()


def from_config_file(config: str, macros: str = None):
    """Instantiate object from config and macros"""
    parsed = parse_config(read_json(config), read_json(macros) if macros else None)
    return from_config(parsed)


def download_config_and_macros_from_mlflow(
    run_id: str,
    path_config: str = None,
    path_macros: str = None,
    tracking_uri: str = None,
    config: str = "config_no_static",
    macros: str = "macros_no_static",
):
    """Download config and macros from MLFlow

    Parameters
    ----------
    run_id : str
        MLFlow Run ID
    path_output : str, optional
        Path output for the configs, if None, current directory
    tracking_uri : str, optional
        MLFlow tracking URI
    config : str, optional
        Name of the config artifact
    macros : str, optional
        Name of the macros artifact
    """
    # Download artifacts from MLFlow
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    path_config_tmp = client.download_artifacts(run_id, f"{config}.json")
    path_macros_tmp = client.download_artifacts(run_id, f"{macros}.json")

    # Copy files to desired location
    path_config = path_config if path_config else "config.json"
    path_macros = path_macros if path_macros else "macros.json"
    copyfile(path_config_tmp, path_config)
    copyfile(path_macros_tmp, path_macros)
    LOGGER.info(f"Downloaded config to '{path_config}'")
    LOGGER.info(f"Downloaded macros to '{path_macros}'")


def add_params(config: str, macros: str, params: Union[List[str], str]):
    """Create new params macro, infer new references in config.

    Parameters
    ----------
    config : str
        Path to config.json
    macros : str
        Path to macros.json
    params : Union[List[str], str]
        List of new parameters

    Raises
    ------
    ValueError
        If one param has no reference in config after adding new refs.
    """
    params = params.split(",") if isinstance(params, str) else params
    config_dict = read_json(config)
    macros_dict = read_json(macros)
    config_dict = add_macro_params(config_dict, macro="params", params=params)
    macros_dict["params"] = {**macros_dict.get("params", {}), **{key: f"$params:{key}" for key in params}}
    write_json(config_dict, config)
    write_json(macros_dict, macros)


def main():
    logging.basicConfig(level=logging.INFO)
    fire.Fire(
        {
            "run": run,
            "from_config_file": from_config_file,
            "download_config_and_macros_from_mlflow": download_config_and_macros_from_mlflow,
            "add_params": add_params,
        }
    )


if __name__ == "__main__":
    main()
