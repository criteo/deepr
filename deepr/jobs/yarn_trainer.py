"""Yarn Trainer Config and Job"""

from abc import ABC
import atexit
import logging
from typing import Dict
from dataclasses import dataclass

import skein
from cluster_pack.packaging import get_editable_requirements
from tf_yarn import run_on_yarn, NodeLabel
from mlflow import end_run

from deepr.config.base import from_config
from deepr.jobs import base
from deepr.jobs.trainer import Trainer


LOGGER = logging.getLogger(__name__)


@dataclass
class YarnTrainerConfig(ABC):
    """Abstract Yarn Trainer Config"""

    name: str = "default-name"
    tf_yarn: str = "cpu"
    queue: str = "default"
    pre_script_hook: str = ""
    nb_retries: int = 0

    @property
    def task_specs(self) -> Dict:
        """Create task specs from config"""
        raise NotImplementedError()

    @property
    def env_vars(self) -> Dict[str, str]:
        """Create environment variables from config"""
        raise NotImplementedError()

    def upload_cpu_env(self):
        """Upload CPU environment to HDFS"""
        raise NotImplementedError()

    def upload_gpu_env(self):
        """Upload GPU environment to HDFS"""
        raise NotImplementedError()


@dataclass
class YarnTrainer(base.Job):
    """Run a `Trainer` job on yarn using distributed settings."""

    trainer: Dict
    config: YarnTrainerConfig

    def __post_init__(self):
        trainer = from_config(self.trainer)
        if not isinstance(trainer, Trainer):
            raise TypeError(f"Expected job of type {Trainer} but got {type(trainer)}")

    def run(self):
        # Upload environment(s) to HDFS (CPU and / or GPU environments)
        pyenv_zip_path = {NodeLabel.CPU: self.config.upload_cpu_env()}
        if self.config.tf_yarn == "gpu":
            pyenv_zip_path[NodeLabel.GPU] = self.config.upload_gpu_env()

        def _experiment_fn():
            # Remove auto-termination of active MLFlow runs from inside
            # the chief / evaluator
            atexit.unregister(end_run)
            return from_config(self.trainer).create_experiment()

        run_on_yarn(
            pyenv_zip_path=pyenv_zip_path,
            experiment_fn=_experiment_fn,
            task_specs=self.config.task_specs,
            files=get_editable_requirements(),
            env=self.config.env_vars,
            queue=self.config.queue,
            pre_script_hook=self.config.pre_script_hook,
            acls=skein.model.ACLs(enable=True, ui_users=["*"], view_users=["*"]),
            nb_retries=self.config.nb_retries,
            name=self.config.name,
        )

        # Run exporters and final evaluation
        trainer = from_config(self.trainer)
        experiment = trainer.create_experiment()
        for exporter in trainer.exporters:
            exporter(experiment.estimator)
        trainer.run_final_evaluation()
