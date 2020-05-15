"""Initializes MLFlow run"""

import logging
import os

from deepr.utils import mlflow


LOGGER = logging.getLogger(__name__)


class MLFlowInit(dict):
    """MLFlow Macro initializes MLFlow run and sets MLFlow parameters"""

    def __init__(
        self,
        use_mlflow: bool = False,
        run_name: str = None,
        tracking_uri: str = None,
        experiment_name: str = None,
        artifact_location: str = None,
    ):
        if use_mlflow:
            # Check arguments are not None
            if run_name is None:
                raise ValueError("'run_name' should not be None")
            if tracking_uri is None:
                raise ValueError("'tracking_uri' should not be None")
            if experiment_name is None:
                raise ValueError("'experiment_name' should not be None")
            if artifact_location is None:
                raise ValueError("'artifact_location' should not be None")

            # Start MLFlow run
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name, artifact_location)
            run = mlflow.start_run(run_name=run_name)

            # Define new parameters
            run_id = run.info.run_id
            run_uuid = run.info.run_uuid
            experiment_id = run.info.experiment_id
            url = f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"

            # MLFlow config in environment variables
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
            os.environ["MLFLOW_RUN_ID"] = run_id

            # Log variables
            LOGGER.info(f"MLFLOW_RUN_UUID: {run_uuid}")
            LOGGER.info(f"MLFLOW_RUN_ID: {run_id}")
            LOGGER.info(f"MLFLOW_TRACKING_URI: {tracking_uri}")
            LOGGER.info(f"MLFLOW_URL: {url}")
        else:
            # Set parameters to None
            run_id = None  # type: ignore
            run_uuid = None  # type: ignore
            experiment_id = None  # type: ignore
            url = None  # type: ignore
            LOGGER.warning("MLFlow not initialized")

        super().__init__(
            url=url,
            run_id=run_id,
            run_uuid=run_uuid,
            run_name=run_name,
            use_mlflow=use_mlflow,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            artifact_location=artifact_location,
            tracking_uri=tracking_uri,
        )
