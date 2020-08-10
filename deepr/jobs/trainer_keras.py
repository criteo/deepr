"""Keras trainer."""

from dataclasses import dataclass, field, fields
from typing import Callable, Dict, List
import logging

import tensorflow as tf
from tf_yarn import Experiment

from deepr.jobs.trainer_base import TrainerBase
from deepr.jobs.trainer import TrainSpec, EvalSpec, FinalSpec, RunConfig, ConfigProto
from deepr.hooks.base import EstimatorHookFactory, TensorHookFactory


LOGGER = logging.getLogger(__name__)


@dataclass
class TrainerKeras(TrainerBase):
    """Keras trainer."""

    # Required arguments
    path_model: str
    model: tf.keras.Model
    train_input_fn: Callable[[], tf.data.Dataset]
    eval_input_fn: Callable[[], tf.data.Dataset]

    # Preprocessing and Exporters
    prepro_fn: Callable[[tf.data.Dataset, str], tf.data.Dataset] = field(default=lambda dataset, _: dataset)
    exporters: List[Callable] = field(default_factory=list)

    # Hooks
    train_hooks: List = field(default_factory=list)
    eval_hooks: List = field(default_factory=list)
    final_hooks: List = field(default_factory=list)

    # Specs and Configs
    train_spec: Dict = field(default_factory=TrainSpec)
    eval_spec: Dict = field(default_factory=EvalSpec)
    final_spec: Dict = field(default_factory=FinalSpec)
    run_config: Dict = field(default_factory=RunConfig)
    config_proto: Dict = field(default_factory=ConfigProto)
    random_seed: int = 42

    def __post_init__(self):
        # Automatically replace None values by the default field value
        for f in fields(self):
            if getattr(self, f.name) is None:
                default = f.default_factory() if callable(f.default_factory) else f.default
                setattr(self, f.name, default)

        # TensorHookFactory are for manual model_fn creation only
        for hooks in [self.train_hooks, self.eval_hooks, self.final_hooks]:
            for hook in hooks:
                if isinstance(hook, TensorHookFactory):
                    raise TypeError(f"{hook} is a TensorHookFactory, not supported by the Keras Trainer.")

    def create_experiment(self):
        """Create an Experiment object packaging Estimator and Specs.

        Returns
        -------
        Experiment (NamedTuple)
            estimator : tf.estimator.Estimator
            train_spec : tf.estimator.TrainSpec
            eval_spec : tf.estimator.EvalSpec
        """
        tf.set_random_seed(self.random_seed)

        # Create Estimator
        LOGGER.info("Converting Keras model to Estimator.")
        model_dir = self.path_model + "/checkpoints"
        estimator = tf.keras.estimator.model_to_estimator(self.model, model_dir=model_dir)

        # Create Hooks
        estimator_train_hooks = [hook(estimator) for hook in self.train_hooks if isinstance(hook, EstimatorHookFactory)]
        estimator_eval_hooks = [hook(estimator) for hook in self.eval_hooks if isinstance(hook, EstimatorHookFactory)]
        train_hooks = [hk for hk in self.train_hooks if not isinstance(hk, (TensorHookFactory, EstimatorHookFactory))]
        eval_hooks = [hk for hk in self.eval_hooks if not isinstance(hk, (TensorHookFactory, EstimatorHookFactory))]

        # Create train specs
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: self.prepro_fn(self.train_input_fn(), tf.estimator.ModeKeys.TRAIN),
            hooks=estimator_train_hooks + train_hooks,
            **self.train_spec,
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: self.prepro_fn(self.eval_input_fn(), tf.estimator.ModeKeys.EVAL),
            hooks=estimator_eval_hooks + eval_hooks,
            **self.eval_spec,
        )
        return Experiment(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)

    def run_final_evaluation(self):
        """Final evaluation on eval_input_fn with final_hooks"""
        # Create Estimator
        LOGGER.info("Converting Keras model to Estimator.")
        model_dir = self.path_model + "/checkpoints"
        estimator = tf.keras.estimator.model_to_estimator(self.model, model_dir=model_dir)

        # Create Hooks
        estimator_final_hooks = [hook(estimator) for hook in self.final_hooks if isinstance(hook, EstimatorHookFactory)]
        final_hooks = [hk for hk in self.final_hooks if not isinstance(hk, (TensorHookFactory, EstimatorHookFactory))]

        # Evaluate final metrics
        global_step = estimator.get_variable_value("global_step")
        LOGGER.info(f"Running final evaluation, using global_step = {global_step}")
        final_metrics = estimator.evaluate(
            lambda: self.prepro_fn(self.eval_input_fn(), tf.estimator.ModeKeys.EVAL),
            hooks=estimator_final_hooks + final_hooks,
            **self.final_spec,
        )
        LOGGER.info(final_metrics)
