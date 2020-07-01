"""Train Job"""

from dataclasses import dataclass, field, fields
import functools
from typing import Callable, Dict, List, Tuple, Iterable
import logging

import tensorflow as tf
from tf_yarn import Experiment

from deepr.jobs.base import Job
from deepr.hooks.base import EstimatorHookFactory, TensorHookFactory


LOGGER = logging.getLogger(__name__)


class TrainSpec(dict):
    """Named Dict for TrainSpec arguments with reasonable defaults."""

    def __init__(self, max_steps: int = None):
        super().__init__(max_steps=max_steps)


class EvalSpec(dict):
    """Named Dict for EvalSpec arguments with reasonable defaults."""

    def __init__(self, steps: int = None, name: str = None, start_delay_secs: int = 120, throttle_secs: int = 100):
        super().__init__(steps=steps, name=name, start_delay_secs=start_delay_secs, throttle_secs=throttle_secs)


class ConfigProto(dict):
    """Named Dict for ConfigProto arguments with reasonable defaults."""

    def __init__(
        self,
        inter_op_parallelism_threads: int = 16,
        intra_op_parallelism_threads: int = 16,
        log_device_placement: bool = False,
        gpu_device_count: int = 0,
        cpu_device_count: int = 16,
        **kwargs,
    ):
        super().__init__(
            inter_op_parallelism_threads=inter_op_parallelism_threads,
            intra_op_parallelism_threads=intra_op_parallelism_threads,
            log_device_placement=log_device_placement,
            device_count={"GPU": gpu_device_count, "CPU": cpu_device_count},
            **kwargs,
        )


class FinalSpec(dict):
    """Named Dict for final evaluation with reasonable defaults."""


class RunConfig(dict):
    """Named Dict for RunConfig arguments"""


@dataclass
class Trainer(Job):
    """Train and evaluate a tf.Estimator on the current machine.

    Attributes
    ----------
    path_model : str
        Path to the model directory. Can be either local or HDFS.
    pred_fn : Callable[[Dict[str, tf.Tensor], str], Dict[str, tf.Tensor]]
        Typically a :class:`~deepr.layers.Layer` instance, but in general, any callable.

        Its signature is the following:
          - features : Dict
                Features, yielded by the dataset
          - predictions : Dict
                Predictions

    loss_fn : Callable[[Dict[str, tf.Tensor], str], Dict[str, tf.Tensor]]
        Typically a :class:`~deepr.layers.Layer` instance, but in general, any callable.

        Its signature is the following:
          - features_and_predictions : Dict
                Features and predictions combined
          - losses : Dict
                Losses and metrics

        The value for key "loss" from the output dictionary is then fed
        to the `optimizer_fn`.

    optimizer_fn : Callable[[Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]
        Typically an :class:`~deepr.optimizers.Optimizer` instance, but in general, any callable.

        Its signature is the following:
          - inputs : Dict[str, tf.Tensor]
                Typically has key "loss"`
          - outputs : Dict[str, tf.Tensor]
                Need key "train_op"

    train_input_fn : Callable[[], tf.data.Dataset]
        Typically a :class:`~deepr.readers.Reader` instance, but in general, any callable.

        Used for training.

        Its signature is the following:
            - outputs : tf.data.Dataset
                A newly created dataset. Each call to the input_fn
                should create a new dataset and a new graph.

    eval_input_fn : Callable[[], tf.data.Dataset]
        Typically a :class:`~deepr.readers.Reader` instance, but in general, any callable.

        Used for evaluation.

        Its signature is the following:
            - outputs : tf.data.Dataset
                A newly created dataset. Each call to the input_fn
                should create a new dataset and a new graph.

    prepro_fn: Callable[[tf.data.Dataset, str], tf.data.Dataset], Optional
        Typically a :class:`~deepr.prepros.Prepro` instance, but in general, any callable.

        Its signature is the following:
          - inputs :
              dataset : tf.data.Dataset
                  Created by `train_input_fn` or `eval_input_fn`.
              mode : str
                  One of tf.estimator.ModeKeys.TRAIN, PREDICT or EVAL
          - outputs : tf.data.Dataset
                The preprocessed dataset

    initializer_fn: Callable[[], None], Optional
        Any Callable that sets up initialization by adding an op to the
        default Graph.

    train_metrics: List[Callable], Optional
        Typically, :class:`~deepr.metrics.Metric` instances, but in general, any callables.

        Used for training.

        Each callable must have the following signature:
          - inputs : Dict
                Features, Predictions and Losses dictionary
          - outputs : Dict[str, Tuple]
                Dictionary of tuples of (tensor_value, update_op).

    eval_metrics: List[Callable], Optional
        Typically, :class:`~deepr.metrics.Metric` instances, but in general, any callables.

        Used for evaluation.

        Each callable must have the following signature:
          - inputs : Dict
                Features, Predictions and Losses dictionary
          - outputs : Dict[str, Tuple]
                Dictionary of tuples of (tensor_value, update_op).

    exporters: List[Callable], Optional
        Typically, :class:`~deepr.exporters.Exporter` instances, but in general, any callables.

        Used at the end of training on the trained :mod:`~`tf.Estimator`.

        Each callable must have the following signature:
          - inputs : tf.estimator.Estimator
                A trained Estimator.

    train_hooks: List, Optional
        List of `Hooks` or `HookFactories`.

        Used for training.

        Some hook can be fully defined during instantiation of Trainer,
        for example a :class:`~deepr.hooks.StepsPerSecHook`. However, other hooks requires
        objects to be instantiated that will only be created after
        running the :class:`~deepr.jobs.Trainer`.

        The `hooks` module defines factories for more complicated hooks.

    eval_hooks: List, Optional
        List of `Hooks` or `HookFactories`.

        Used for evaluation.

        Some hook can be fully defined during instantiation of Trainer,
        for example a :class:`~deepr.hooks.StepsPerSecHook`. However, other hooks requires
        objects to be instantiated that will only be created after
        running the :class:`~deepr.jobs.Trainer`.

        The `hooks` module defines factories for more complicated hooks.

    eval_spec: Dict, Optional
        Optional parameters for :class:`~tf.estimator.EvalSpec`.
    train_spec: Dict, Optional
        Optional parameters for :class:`~tf.estimator.TrainSpec`.
    run_config: Dict, Optional
        Optional parameters for :class:`~tf.estimator.RunConfig`.
    config_proto: Dict, Optional
        Optional parameters for :class:`~tf.estimator.RunConfig`.
    """

    path_model: str
    pred_fn: Callable[[Dict[str, tf.Tensor], str], Dict[str, tf.Tensor]]
    loss_fn: Callable[[Dict[str, tf.Tensor], str], Dict[str, tf.Tensor]]
    optimizer_fn: Callable[[Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]
    train_input_fn: Callable[[], tf.data.Dataset]
    eval_input_fn: Callable[[], tf.data.Dataset]

    # Optional Arguments
    prepro_fn: Callable[[tf.data.Dataset, str], tf.data.Dataset] = field(default=lambda dataset, _: dataset)
    initializer_fn: Callable[[], None] = field(default=lambda: None)
    exporters: List[Callable] = field(default_factory=list)
    train_metrics: List[Callable] = field(default_factory=list)
    eval_metrics: List[Callable] = field(default_factory=list)
    final_metrics: List[Callable] = field(default_factory=list)
    train_hooks: List = field(default_factory=list)
    eval_hooks: List = field(default_factory=list)
    final_hooks: List = field(default_factory=list)
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

    def run(self):
        """Train, evaluate and export Estimator"""
        experiment = self.create_experiment()
        tf.estimator.train_and_evaluate(experiment.estimator, experiment.train_spec, experiment.eval_spec)
        for exporter in self.exporters:
            exporter(experiment.estimator)
        self.run_final_evaluation()

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
        model_dir = self.path_model + "/checkpoints"
        estimator = tf.estimator.Estimator(
            functools.partial(
                model_fn,
                pred_fn=self.pred_fn,
                loss_fn=self.loss_fn,
                optimizer_fn=self.optimizer_fn,
                initializer_fn=self.initializer_fn,
                train_metrics=self.train_metrics,
                eval_metrics=self.eval_metrics,
                train_hooks=[hook for hook in self.train_hooks if isinstance(hook, TensorHookFactory)],
                eval_hooks=[hook for hook in self.eval_hooks if isinstance(hook, TensorHookFactory)],
            ),
            model_dir=model_dir,
            config=tf.estimator.RunConfig(
                session_config=tf.ConfigProto(**self.config_proto), model_dir=model_dir, **self.run_config
            ),
        )

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
        model_dir = self.path_model + "/checkpoints"
        estimator = tf.estimator.Estimator(
            functools.partial(
                model_fn,
                pred_fn=self.pred_fn,
                loss_fn=self.loss_fn,
                optimizer_fn=self.optimizer_fn,
                initializer_fn=self.initializer_fn,
                train_metrics=self.train_metrics,
                eval_metrics=self.final_metrics,
                train_hooks=[hook for hook in self.train_hooks if isinstance(hook, TensorHookFactory)],
                eval_hooks=[hook for hook in self.final_hooks if isinstance(hook, TensorHookFactory)],
            ),
            model_dir=model_dir,
        )

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


def model_fn(
    features: Dict[str, tf.Tensor],
    mode: tf.estimator.ModeKeys,
    pred_fn: Callable[[Dict[str, tf.Tensor], str], Dict[str, tf.Tensor]],
    loss_fn: Callable[[Dict[str, tf.Tensor], str], Dict[str, tf.Tensor]],
    optimizer_fn: Callable[[tf.Tensor], tf.Tensor],
    initializer_fn: Callable[[], None],
    train_metrics: Iterable[Callable],
    eval_metrics: Iterable[Callable],
    train_hooks: Iterable,
    eval_hooks: Iterable,
):
    """Model Function"""
    predictions = pred_fn(features, mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    losses = loss_fn({**features, **predictions}, mode)
    loss = losses["loss"]
    initializer_fn()

    metrics = {}  # type: Dict[str, Tuple]
    if mode == tf.estimator.ModeKeys.EVAL:
        for metric_fn in eval_metrics:
            metrics.update(metric_fn({**features, **predictions, **losses}))
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops={key if key != "loss" else "average_loss": metric for key, metric in metrics.items()},
            evaluation_hooks=[hook({key: val for key, (val, _) in metrics.items()}) for hook in eval_hooks],
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        updates = optimizer_fn({**features, **predictions, **losses})
        for metric_fn in train_metrics:
            metrics.update(metric_fn({**features, **predictions, **losses, **updates}))
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=tf.group(updates["train_op"], *[op for _, op in metrics.values()]),
            training_hooks=[hook({key: val for key, (val, _) in metrics.items()}) for hook in train_hooks],
        )

    raise RuntimeError(f"Mode {mode} is not supported")
