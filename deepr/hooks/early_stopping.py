"""Early Stopping Hook"""

import logging
from functools import partial
from typing import Callable, Union

import tensorflow as tf

from deepr.exporters.best_checkpoint import BestMode, read_eval_metrics
from deepr.hooks.base import EstimatorHookFactory


LOGGER = logging.getLogger(__name__)


class EarlyStoppingHookFactory(EstimatorHookFactory):
    """Early Stopping Hook Factory

    Attributes
    ----------
    metric : str
        Name of the metric to read from evaluation checkpoints.
    max_steps_without_improvement : int
        If no improvement in metric for this many step, stop.
    min_steps : int
        Do not attempt to early stop for this many steps.
    mode : BestMode
        INCREASE or DECREASE. If DECREASE, lower metric is better.
    run_every_secs : int, Optional
        If given, run early stopping hook every given seconds.
    run_every_steps : int, Optional
        If given, run early stopping hook every given steps.

        Either `run_every_secs` or `run_every_step` should be given.
    final_step : int, Optional
        If given, will set the global_step to this value when early
        stopping.

        This is a useful way to signal the end of training to the
        evaluator in the case of distributed training (early stopping
        causes issues with `train_and_evaluate`).
    """

    def __init__(
        self,
        metric: str,
        max_steps_without_improvement: int,
        min_steps: int = 0,
        mode: Union[str, BestMode] = BestMode.DECREASE,
        run_every_secs: int = None,
        run_every_steps: int = None,
        final_step: int = None,
    ):
        if run_every_secs is None and run_every_steps is None:
            raise ValueError("run_every_steps and run_every_secs cannot both be None.")
        if isinstance(mode, str):
            mode = BestMode(mode)
        if not isinstance(mode, BestMode):
            raise ValueError(f"Expected {BestMode} but got {mode}")

        self.metric = metric
        self.max_steps_without_improvement = max_steps_without_improvement
        self.mode = mode
        self.min_steps = min_steps
        self.run_every_secs = run_every_secs
        self.run_every_steps = run_every_steps
        self.final_step = final_step

    def __call__(self, estimator: tf.estimator.Estimator) -> tf.estimator.SessionRunHook:
        if estimator.config.is_chief:
            return _StopOnPredicateHook(
                partial(
                    _no_metric_improvement_fn,
                    eval_dir=estimator.eval_dir(),
                    min_steps=self.min_steps,
                    metric=self.metric,
                    max_steps_without_improvement=self.max_steps_without_improvement,
                    mode=self.mode,
                ),
                run_every_secs=self.run_every_secs,
                run_every_steps=self.run_every_steps,
                final_step=self.final_step,
            )
        else:
            return _CheckForStoppingHook()


def _no_metric_improvement_fn(
    global_step: int,
    eval_dir: str,
    min_steps: int,
    metric: str,
    max_steps_without_improvement: int,
    mode: BestMode = BestMode.DECREASE,
):
    """Returns `True` if metric does not improve within max steps."""
    if global_step < min_steps:
        return False

    is_better_fn = {
        BestMode.INCREASE: lambda val, best_val: val > best_val,
        BestMode.DECREASE: lambda val, best_val: val < best_val,
    }
    eval_results = read_eval_metrics(eval_dir)
    best_val, best_val_step = None, None
    for step, metrics in eval_results.items():
        val = metrics[metric]
        if best_val is None or is_better_fn[mode](val, best_val):
            best_val, best_val_step = val, step
        if step - best_val_step >= max_steps_without_improvement:
            msg = f"No {mode} in metric {metric} for {step - best_val_step} steps, which is greater than or equal "
            msg += f"to max steps ({max_steps_without_improvement}) configured for early stopping."
            LOGGER.info(msg)
            return True
    return False


def _get_or_create_stop_var():
    with tf.variable_scope(name_or_scope="signal_early_stopping", values=[], reuse=tf.AUTO_REUSE):
        return tf.get_variable(
            name="STOP",
            shape=[],
            dtype=tf.bool,
            initializer=tf.constant_initializer(False),
            collections=[tf.GraphKeys.GLOBAL_VARIABLES],
            trainable=False,
        )


class _StopOnPredicateHook(tf.estimator.SessionRunHook):
    """Request stop when should_stop_fn returns True"""

    def __init__(
        self,
        should_stop_fn: Callable[[int], bool],
        run_every_secs: int = None,
        run_every_steps: int = None,
        final_step: int = None,
    ):
        self._should_stop_fn = should_stop_fn
        self._timer = tf.estimator.SecondOrStepTimer(every_secs=run_every_secs, every_steps=run_every_steps)
        self._global_step_tensor = None
        self._stop_var = None
        self._stop_op = None
        self._final_step = final_step

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        self._stop_var = _get_or_create_stop_var()
        self._stop_op = tf.assign(self._stop_var, True)
        self._final_step_op = tf.assign(self._global_step_tensor, self._final_step) if self._final_step else None

    def before_run(self, run_context):
        del run_context
        return tf.estimator.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        """Request stop and override global_step if should stop"""
        global_step = run_values.results
        if self._timer.should_trigger_for_step(global_step):
            self._timer.update_last_triggered_step(global_step)
            if self._should_stop_fn(global_step):
                LOGGER.info(f"Requesting early stopping at global step {global_step}")
                if self._final_step:
                    LOGGER.info(f"Updating global step to {self._final_step} to force evaluator to stop")
                    run_context.session.run(self._final_step_op)
                run_context.session.run(self._stop_op)
                run_context.request_stop()


class _CheckForStoppingHook(tf.estimator.SessionRunHook):
    """Request stop if stop requested by _StopOnPredicateHook"""

    def __init__(self):
        self._stop_var = None

    def begin(self):
        self._stop_var = _get_or_create_stop_var()

    def before_run(self, run_context):
        del run_context
        return tf.estimator.SessionRunArgs(self._stop_var)

    def after_run(self, run_context, run_values):
        should_early_stop = run_values.results
        if should_early_stop:
            LOGGER.info("Early stopping requested, suspending run.")
            run_context.request_stop()
