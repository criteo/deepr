"""Tensorflow Optimizers"""

from typing import Dict, List
import logging

import tensorflow as tf

from deepr.optimizers import base


LOGGER = logging.getLogger(__name__)


class TensorflowOptimizer(base.Optimizer):
    """Default Tensorflow Optimizers

    Attributes
    ----------
    learning_rate : float
        Learning rate
    optimizer : str
        Name of the optimizer. See `TensorflowOptimizer.OPTIMIZERS` for
        a description of available Tensorflow optimizers.
    kwargs :
        Optional arguments for the Tensorflow optimizer.
    """

    OPTIMIZERS = {
        "adam": tf.train.AdamOptimizer,
        "adagrad": tf.train.AdagradOptimizer,
        "lazyadam": tf.contrib.opt.LazyAdamOptimizer,
        "sgd": tf.train.GradientDescentOptimizer,
        "momentum": tf.train.MomentumOptimizer,
    }

    def __init__(
        self,
        optimizer: str,
        learning_rate: float,
        loss: str = "loss",
        grad_norms: List[str] = None,
        exclude_vars: List[str] = None,
        clip: float = None,
        skip_vars: List[str] = None,
        skip_steps: int = None,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.grad_norms = grad_norms
        self.exclude_vars = exclude_vars
        self.clip = clip
        self.skip_vars = skip_vars
        self.skip_steps = skip_steps
        self._kwargs = kwargs
        if not self.optimizer.lower() in self.OPTIMIZERS:
            raise ValueError(f"Optimizer {self.optimizer} not in {self.OPTIMIZERS}")

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict:
        # Compute gradients
        loss = tensors[self.loss]
        optimizer = self.OPTIMIZERS[self.optimizer.lower()](learning_rate=self.learning_rate, **self._kwargs)
        grad_and_vars = optimizer.compute_gradients(loss)

        # Exclude variables
        if self.exclude_vars is not None:
            filtered_grad_and_vars = []
            for grad, var in grad_and_vars:
                if any(name in var.name for name in self.exclude_vars):
                    LOGGER.info(f"Not training {var.name}")
                    continue
                filtered_grad_and_vars.append((grad, var))
            grad_and_vars = filtered_grad_and_vars

        # Clip gradients by global norm
        if self.clip is not None:
            grads, variables = zip(*grad_and_vars)
            grads, _ = tf.clip_by_global_norm(grads, self.clip)
            grad_and_vars = zip(grads, variables)

        # Train op on all variables
        train_op = optimizer.apply_gradients(grad_and_vars, global_step=tf.train.get_global_step())

        # Train op that skip some variables for a few steps
        if self.skip_vars and self.skip_steps:
            non_skipped_grad_and_vars = []
            for grad, var in grad_and_vars:
                if any(name in var.name for name in self.skip_vars):
                    LOGGER.info(f"Skip training of {var.name} for {self.skip_steps} steps")
                    continue
                non_skipped_grad_and_vars.append((grad, var))
            non_skipped_train_op = optimizer.apply_gradients(
                non_skipped_grad_and_vars, global_step=tf.train.get_global_step()
            )
            train_op = tf.cond(
                tf.train.get_global_step() <= self.skip_steps, lambda: non_skipped_train_op, lambda: train_op,
            )

        # Compute gradients global norm for names in self.grad_norms
        grad_norms = dict()
        if self.grad_norms is not None:
            for grad, var in grad_and_vars:
                if any(name in var.name for name in self.grad_norms):
                    if self.skip_vars and self.skip_steps and any(name in var.name for name in self.skip_vars):
                        norm = tf.cond(
                            tf.train.get_global_step() <= self.skip_steps, lambda: 0.0, lambda: tf.global_norm([grad])
                        )
                    else:
                        norm = tf.global_norm([grad])
                    grad_norms[f"grad_norm_{var.name}"] = norm

        return {**grad_norms, "train_op": train_op}
