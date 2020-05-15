"""MLFlow utilities."""

from pathlib import Path
import logging
from typing import Dict
import json
import tempfile

import mlflow
from mlflow.utils import env
from mlflow.tracking import fluent

from deepr.utils.exceptions import handle_exceptions


LOGGER = logging.getLogger(__name__)

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
is_tracking_uri_set = handle_exceptions(mlflow.is_tracking_uri_set)
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
start_run = handle_exceptions(mlflow.start_run)
update_artifacts_location = handle_exceptions(mlflow.update_artifacts_location)


@handle_exceptions
def set_experiment(name: str, artifact_location: str = None):
    """Set Experiment with specific artifact_location."""
    if mlflow.tracking.client.MlflowClient().get_experiment_by_name(name) is None:
        mlflow.create_experiment(name=name, artifact_location=artifact_location)
    else:
        mlflow.set_experiment(name)


@handle_exceptions
def log_dict(data: Dict, filename: str):
    """Log dictionary to MLFlow as an artifact under filename."""
    path = Path(tempfile.mkdtemp(), filename)
    with path.open("w") as file:
        json.dump(data, file, indent=4)
    mlflow.log_artifact(local_path=str(path), artifact_path="")


@handle_exceptions
def clear_run():
    """Clear run (remove from MLFlow stack and unset ENV variable)."""
    # pylint: disable=protected-access
    if len(fluent._active_run_stack) > 0:
        env.unset_variable("MLFLOW_RUN_ID")
        fluent._active_run_stack.pop()
