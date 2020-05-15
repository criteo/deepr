"""Best Checkpoint Exporter"""

from collections import defaultdict
from enum import Enum
import re
import logging
from typing import Dict, Union

import tensorflow as tf

from deepr.exporters import base
from deepr.io.path import Path
from deepr.utils import mlflow


LOGGER = logging.getLogger(__name__)

_CHEKPOINT_PATTERN = "*model.ckpt-*.index"

_SUMMARY_PATTERN = "*.tfevents.*"


class BestMode(Enum):
    """Modes for early stopping"""

    INCREASE: str = "increase"
    DECREASE: str = "decrease"


class BestCheckpoint(base.Exporter):
    """Overrides Checkpoint Information to point to the best checkpoint.

    The best checkpoint is the one whose step is closest to the step of
    the best evaluation metric. The evaluation metrics are retrieved
    from the evaluation summaries (located in `eval`).

    Attributes
    ----------
    metric : str
        Name of the metric used to select the best checkpoint
    mode : str
        If 'decrease', means lower is better.
    tag : str
        Tag for MLFlow
    use_mlflow : bool
        If True, will log the best checkpoint step as tag to MLFlow
    """

    def __init__(
        self,
        metric: str,
        mode: Union[str, BestMode] = BestMode.DECREASE,
        use_mlflow: bool = False,
        tag: str = "training_best_step",
    ):
        if isinstance(mode, str):
            mode = BestMode(mode)
        self.metric = metric
        self.mode = mode
        self.use_mlflow = use_mlflow
        self.tag = tag

    def export(self, estimator: tf.estimator.Estimator):
        # Reload summaries and select best step
        LOGGER.info(f"Reloading summaries from {estimator.model_dir}")
        summaries = read_eval_metrics(estimator.eval_dir()).items()
        for step, metrics in sorted(summaries):
            LOGGER.info(f"- {step}: {metrics}")
        sorted_summaries = sorted(summaries, key=lambda t: t[1][self.metric])
        if self.mode == BestMode.INCREASE:
            best_step, best_metrics = sorted_summaries[-1]
        elif self.mode == BestMode.DECREASE:
            best_step, best_metrics = sorted_summaries[0]
        else:
            raise ValueError(f"Mode {self.mode} not recognized.")
        LOGGER.info(f"Best summary at step {best_step}: {best_metrics}")

        # List available checkpoints and select closes to best_step
        checkpoints = Path(estimator.model_dir).glob(_CHEKPOINT_PATTERN)
        checkpoint_steps = [int(re.findall(r"-(\d+).index", str(path))[0]) for path in checkpoints]
        selected_step = sorted(checkpoint_steps, key=lambda step: abs(step - best_step))[0]
        LOGGER.info(f"Selected checkpoint {selected_step}")

        # Change checkpoint information
        with Path(estimator.model_dir, "checkpoint").open("r") as file:
            lines = file.read().split("\n")
            lines[0] = f'model_checkpoint_path: "model.ckpt-{selected_step}"'

        with Path(estimator.model_dir, "checkpoint").open("w") as file:
            file.write("\n".join(lines))

        # Check that change is effective
        global_step = estimator.get_variable_value("global_step")
        if not global_step == selected_step:
            msg = f"Changed checkpoint file to use step {selected_step}, but estimator uses {global_step}"
            raise ValueError(msg)

        # Log to MLFlow
        if self.use_mlflow:
            mlflow.log_metric(key=self.tag, value=global_step)


def read_eval_metrics(eval_dir: str) -> Dict:
    """Reload summaries from model_dir"""
    if not Path(eval_dir).is_dir():
        return dict()
    summaries = defaultdict(dict)  # type: Dict[int, Dict[str, float]]
    for path in Path(eval_dir).glob(_SUMMARY_PATTERN):
        for event in tf.train.summary_iterator(str(path)):
            if not event.HasField("summary"):
                continue
            metrics = {}
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    metrics[value.tag] = value.simple_value
            if metrics:
                summaries[event.step].update(metrics)
    return summaries
