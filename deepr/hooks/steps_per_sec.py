"""Steps Per Second Hook"""

import logging

import tensorflow as tf

from deepr.utils import mlflow
from deepr.utils import graphite


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
        name: str = None,
        use_mlflow: bool = False,
        use_graphite: bool = False,
        skip_after_step: int = None,
        every_n_steps: int = 100,
        every_n_secs: int = None,
        output_dir: str = None,
        summary_writer=None,
    ):
        super().__init__(
            every_n_steps=every_n_steps, every_n_secs=every_n_secs, output_dir=output_dir, summary_writer=summary_writer
        )
        self.batch_size = batch_size
        self.name = name
        self.use_mlflow = use_mlflow
        self.use_graphite = use_graphite
        self.skip_after_step = skip_after_step

    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        """Log Steps per second and write summary"""
        if self.skip_after_step is not None and global_step >= self.skip_after_step:
            return

        # Compute steps and number of examples per second
        metrics = {
            "steps_per_sec": elapsed_steps / elapsed_time,
            "examples_per_sec": self.batch_size * elapsed_steps / elapsed_time,
        }

        # Log tensor values
        LOGGER.info(", ".join(f"{tag} = {value:.2f}" for tag, value in metrics.items()))

        # Send to MLFlow and Graphite
        for tag, value in metrics.items():
            if self.use_graphite:
                graphite.log_metric(tag, value, postfix=self.name)
            if self.use_mlflow:
                tag = tag if self.name is None else f"{self.name}_{tag}"
                mlflow.log_metric(tag, value, step=global_step)
