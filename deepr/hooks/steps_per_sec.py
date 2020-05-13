"""Steps Per Second Hook"""

import logging
from typing import Dict

import tensorflow as tf
import mlflow
import graphyte


LOGGER = logging.getLogger(__name__)


class StepsPerSecHook(tf.train.StepCounterHook):
    """Logs steps per seconds and num_examples_per_sec.

    Attributes
    ----------
    batch_size : int
        Batch Size
    prefix: str, Optional
        Prefix of tags when sending to MLFlow / Graphite
    use_mlflow: bool, Optional
        If True, send metrics to MLFlow. Default is False.
    use_graphite: bool, Optional
        If True, send metrics to Graphite. Default is False.
    skip_after_step: int, Optional
        If not None, do not run the hooks after this step.
    """

    def __init__(
        self,
        batch_size: int,
        prefix: str = "",
        use_mlflow: bool = False,
        use_graphite: bool = False,
        config_graphite: Dict = None,
        skip_after_step: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.prefix = prefix
        self.use_mlflow = use_mlflow
        self.use_graphite = use_graphite
        self.config_graphite = config_graphite
        self.skip_after_step = skip_after_step

        self._graphite_sender = graphyte.Sender(**config_graphite) if self.use_graphite else None

    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        """Log Steps per second and write summary"""
        if self.skip_after_step is not None and global_step >= self.skip_after_step:
            return
        steps_per_sec = elapsed_steps / elapsed_time
        examples_per_sec = self.batch_size * steps_per_sec
        LOGGER.info(f"steps_per_sec = {steps_per_sec:.2f}, examples_per_sec = {examples_per_sec:.2f}")
        if self.use_mlflow:
            mlflow.log_metric(f"{self.prefix}steps_per_sec", value=steps_per_sec, step=global_step)
            mlflow.log_metric(f"{self.prefix}examples_per_sec", value=examples_per_sec, step=global_step)
        if self.use_graphite:
            self._graphite_sender.send(f"{self.prefix}steps_per_sec", steps_per_sec)
            self._graphite_sender.send(f"{self.prefix}examples_per_sec", examples_per_sec)
