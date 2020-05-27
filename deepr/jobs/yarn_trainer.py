"""Yarn Trainer Config and Job"""

from dataclasses import dataclass
from typing import Dict, Optional
import atexit
import datetime
import logging

from cluster_pack.packaging import get_editable_requirements
import mlflow
import skein
import tf_yarn

from deepr.config.base import from_config
from deepr.jobs import base
from deepr.jobs.yarn_config import YarnConfig


LOGGER = logging.getLogger(__name__)


@dataclass
class YarnTrainerConfig(YarnConfig):
    """Default Yarn Trainer Config"""

    name: str = f"yarn-trainer-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    nb_ps: Optional[int] = None
    nb_retries: int = 0
    nb_workers: Optional[int] = None

    pre_script_hook: str = "source /etc/profile.d/cuda.sh && setupCUDA 10.1 && " "setupCUDNN cuda10.1_v7.6.4.38"
    queue: str = "dev"

    tf_yarn: str = "cpu"
    tf_yarn_chief_cores: int = 48
    tf_yarn_chief_memory: str = "48 GiB"
    tf_yarn_evaluator_cores: int = 48
    tf_yarn_evaluator_memory: str = "48 GiB"
    tf_yarn_tensorboard_memory: str = "48 GiB"

    def get_task_specs(self):
        """Return Task Specs from parameters"""
        label = tf_yarn.NodeLabel.CPU if self.tf_yarn == "cpu" else tf_yarn.NodeLabel.GPU
        specs = {
            "chief": tf_yarn.TaskSpec(memory=self.tf_yarn_chief_memory, vcores=self.tf_yarn_chief_cores, label=label),
            "evaluator": tf_yarn.TaskSpec(memory=self.tf_yarn_evaluator_memory, vcores=self.tf_yarn_evaluator_cores),
            "tensorboard": tf_yarn.TaskSpec(memory=self.tf_yarn_tensorboard_memory, vcores=8),
        }
        if self.nb_workers is not None:
            specs["worker"] = tf_yarn.TaskSpec(memory="32 GiB", vcores=48, instances=self.nb_workers, label=label)
        if self.nb_ps is not None:
            specs["ps"] = tf_yarn.TaskSpec(memory="32 GiB", vcores=48, instances=self.nb_ps)
        return specs


@dataclass
class YarnTrainer(base.Job):
    """Run a :class:`~deepr.jobs.Trainer` job on yarn using distributed settings."""

    trainer: Dict
    config: YarnTrainerConfig
    train_on_yarn: bool = True

    def run(self):
        if self.train_on_yarn:
            # Upload environment(s) to HDFS (CPU and / or GPU environments)
            pyenv_zip_path = {tf_yarn.NodeLabel.CPU: self.config.upload_pex_cpu()}
            if self.config.tf_yarn == "gpu":
                pyenv_zip_path[tf_yarn.NodeLabel.GPU] = self.config.upload_pex_gpu()

            def _experiment_fn():
                # Remove auto-termination of active MLFlow runs from
                # inside the chief / evaluator
                atexit.unregister(mlflow.end_run)
                return from_config(self.trainer).create_experiment()

            tf_yarn.run_on_yarn(
                acls=skein.model.ACLs(enable=True, ui_users=["*"], view_users=["*"]),
                env=self.config.get_env_vars(),
                experiment_fn=_experiment_fn,
                files=get_editable_requirements(),
                name=self.config.name,
                nb_retries=self.config.nb_retries,
                pre_script_hook=self.config.pre_script_hook,
                pyenv_zip_path=pyenv_zip_path,
                queue=self.config.queue,
                task_specs=self.config.get_task_specs(),
            )

            # Run exporters and final evaluation
            trainer = from_config(self.trainer)
            experiment = trainer.create_experiment()
            for exporter in trainer.exporters:
                exporter(experiment.estimator)
            trainer.run_final_evaluation()
        else:
            LOGGER.info("Not training on yarn.")
            trainer = from_config(self.trainer)
            trainer.run()
