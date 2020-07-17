"""MLFlow utilities."""

from pathlib import Path
import logging
from typing import Dict
import json
import tempfile
import requests
import shutil

import mlflow
from mlflow.utils import env
from mlflow.tracking import fluent

from deepr.utils.exceptions import handle_exceptions


LOGGER = logging.getLogger(__name__)

RUN_ID_ENV_VAR = "MLFLOW_RUN_ID"

TRACKING_URI_ENV_VAR = "MLFLOW_TRACKING_URI"


active_run = handle_exceptions(mlflow.active_run)
create_experiment = handle_exceptions(mlflow.create_experiment)
delete_experiment = handle_exceptions(mlflow.delete_experiment)
delete_run = handle_exceptions(mlflow.delete_run)
delete_tag = handle_exceptions(mlflow.delete_tag)
end_run = handle_exceptions(mlflow.end_run)
get_artifact_uri = handle_exceptions(mlflow.get_artifact_uri)
get_experiment = handle_exceptions(mlflow.get_experiment)
get_experiment_by_name = handle_exceptions(mlflow.get_experiment_by_name)
get_run = handle_exceptions(mlflow.get_run)
get_tracking_uri = handle_exceptions(mlflow.get_tracking_uri)
log_artifact = handle_exceptions(mlflow.log_artifact)
log_artifacts = handle_exceptions(mlflow.log_artifacts)
log_metric = handle_exceptions(mlflow.log_metric)
log_metrics = handle_exceptions(mlflow.log_metrics)
log_param = handle_exceptions(mlflow.log_param)
log_params = handle_exceptions(mlflow.log_params)
register_model = handle_exceptions(mlflow.register_model)
run = handle_exceptions(mlflow.run)
search_runs = handle_exceptions(mlflow.search_runs)
set_tag = handle_exceptions(mlflow.set_tag)
set_tags = handle_exceptions(mlflow.set_tags)
set_tracking_uri = handle_exceptions(mlflow.set_tracking_uri)
set_experiment = handle_exceptions(mlflow.set_experiment)
start_run = handle_exceptions(mlflow.start_run)


@handle_exceptions
def set_or_create_experiment(name: str, artifact_location: str = None):
    """Set Experiment with specific artifact_location."""
    # Set or Create Experiment
    if get_experiment_by_name(name) is None:
        create_experiment(name=name, artifact_location=artifact_location)
    else:
        set_experiment(name)

    # Check Experiment
    experiment = get_experiment_by_name(name)
    LOGGER.info(f"Experiment: {experiment}")
    if artifact_location:
        if experiment.artifact_location != artifact_location:
            msg = f"Incoherent artifact locations. Existing: {experiment.artifact_location}, Got: {artifact_location}"
            LOGGER.warning(msg)


@handle_exceptions
def log_dict(data: Dict, filename: str):
    """Log dictionary to MLFlow as an artifact under filename."""
    path = Path(tempfile.mkdtemp(), filename)
    with path.open("w") as file:
        json.dump(data, file, indent=4)
    mlflow.log_artifact(local_path=str(path), artifact_path="")


@handle_exceptions
def download_artifacts(run_id: str, path: str, dst_path: str = None, tracking_uri: str = None):
    """Download artifacts from MLFlow over HTTP if possible.

    Parameters
    ----------
    run_id : str
        MLFlow Run Id
    path : str
        Path of the artifact
    dst_path : str, optional
        Path on the local system
    """
    # pylint: disable=broad-except
    try:
        tracking_uri = tracking_uri if tracking_uri is not None else get_tracking_uri()
        dst_path = dst_path if dst_path else path
        url = f"{tracking_uri}/get-artifact?path={path}&run_uuid={run_id}"
        LOGGER.info(f"Downloading artifact {path} for run {run_id} over HTTP")
        with Path(dst_path).open("wb") as file:
            file.write(requests.get(url).content)
        LOGGER.info(f"Downloaded artifact {path} to {dst_path}")
    except Exception as e:
        LOGGER.warning(f"Download over HTTP with url {url} failed with exception {e}")
        LOGGER.info("Trying with default MlflowClient")
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        path_tmp = client.download_artifacts(run_id, path)
        shutil.copyfile(str(path_tmp), str(dst_path))
        LOGGER.info(f"Downloaded artifact {path} to {dst_path}")


@handle_exceptions
def clear_run():
    """Clear run (remove from MLFlow stack and unset ENV variable)."""
    # pylint: disable=protected-access
    if len(fluent._active_run_stack) > 0:
        env.unset_variable("MLFLOW_RUN_ID")
        fluent._active_run_stack.pop()
