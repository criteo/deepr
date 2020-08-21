"""Evaluate Job."""

from dataclasses import field, dataclass
import functools
from typing import Callable, Dict, List
import logging

import tensorflow as tf

from deepr.jobs import base
from deepr.jobs.trainer import FinalSpec, model_fn
from deepr.hooks.base import EstimatorHookFactory, TensorHookFactory


LOGGER = logging.getLogger(__name__)


@dataclass
class Evaluate(base.Job):
    """Evaluate Job."""

    path_model: str
    pred_fn: Callable[[Dict[str, tf.Tensor], str], Dict[str, tf.Tensor]]
    loss_fn: Callable[[Dict[str, tf.Tensor], str], Dict[str, tf.Tensor]]
    input_fn: Callable[[], tf.data.Dataset]

    # Optional
    prepro_fn: Callable[[tf.data.Dataset, str], tf.data.Dataset] = field(default=lambda dataset, _: dataset)
    metrics: List[Callable] = field(default_factory=list)
    hooks: List = field(default_factory=list)
    spec: Dict = field(default_factory=FinalSpec)

    def run(self):
        model_dir = self.path_model + "/checkpoints"
        estimator = tf.estimator.Estimator(
            functools.partial(
                model_fn,
                pred_fn=self.pred_fn,
                loss_fn=self.loss_fn,
                optimizer_fn=lambda x: x,
                initializer_fn=lambda: None,
                train_metrics=[],
                eval_metrics=self.metrics,
                train_hooks=[],
                eval_hooks=[hook for hook in self.hooks if isinstance(hook, TensorHookFactory)],
            ),
            model_dir=model_dir,
        )

        # Create Hooks
        estimator_hooks = [hook(estimator) for hook in self.hooks if isinstance(hook, EstimatorHookFactory)]
        hooks = [hk for hk in self.hooks if not isinstance(hk, (TensorHookFactory, EstimatorHookFactory))]

        # Evaluate final metrics
        global_step = estimator.get_variable_value("global_step")
        LOGGER.info(f"Running final evaluation, using global_step = {global_step}")
        metrics = estimator.evaluate(
            lambda: self.prepro_fn(self.input_fn(), tf.estimator.ModeKeys.EVAL),
            hooks=estimator_hooks + hooks,
            **self.spec,
        )
        LOGGER.info(metrics)
