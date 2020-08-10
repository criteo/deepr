"""Trainer Base."""

import logging

from abc import abstractmethod

import tensorflow as tf

from deepr.jobs import base


LOGGER = logging.getLogger(__name__)


class TrainerBase(base.Job):
    """Trainer Base."""

    def run(self):
        experiment = self.create_experiment()
        tf.estimator.train_and_evaluate(experiment.estimator, experiment.train_spec, experiment.eval_spec)
        for exporter in self.exporters:
            exporter(experiment.estimator)
        self.run_final_evaluation()

    @abstractmethod
    def create_experiment(self):
        raise NotImplementedError

    @abstractmethod
    def run_final_evaluation(self):
        raise NotImplementedError
