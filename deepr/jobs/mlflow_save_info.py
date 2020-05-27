"""Save MLFlow info to path"""

from dataclasses import dataclass
from typing import Optional
import logging
import json

from deepr.jobs import base
from deepr.io.path import Path


LOGGER = logging.getLogger(__name__)


@dataclass
class MLFlowSaveInfo(base.Job):
    """Save MLFlow info to path"""

    use_mlflow: Optional[bool] = False
    path_mlflow: Optional[str] = None
    run_id: Optional[str] = None
    run_uuid: Optional[str] = None
    experiment_id: Optional[str] = None

    def run(self):
        if self.use_mlflow:
            LOGGER.info(f"Saving MLFlow info to {self.path_mlflow}")
            # Check arguments are not None
            if self.path_mlflow is None:
                raise ValueError("'path_mlflow' should not be None")
            if self.run_id is None:
                raise ValueError("'run_id' should not be None")
            if self.run_uuid is None:
                raise ValueError("'run_uuid' should not be None")
            if self.experiment_id is None:
                raise ValueError("'experiment_id' should not be None")

            # Save info to path
            info = {"run_id": self.run_id, "run_uuid": self.run_uuid, "experiment_id": self.experiment_id}

            # Need to create directory if not HDFS
            if not Path(self.path_mlflow).is_hdfs:
                Path(self.path_mlflow).parent.mkdir(parents=True, exist_ok=True)

            # Dump MLFlow information to path
            with Path(self.path_mlflow).open("w") as file:
                json.dump(info, file, indent=4)
