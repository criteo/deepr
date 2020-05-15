"""Log Metric Job"""

from dataclasses import dataclass
import logging
from typing import Any

from deepr.jobs import base

from deepr.utils import mlflow


LOGGER = logging.getLogger(__name__)


@dataclass
class LogMetric(base.Job):
    """Log Metric job"""

    key: str
    value: Any
    use_mlflow: bool = False

    def run(self):
        LOGGER.info(f"{self.key}: {self.value}")
        if self.use_mlflow:
            mlflow.log_metric(key=self.key, value=self.value)
