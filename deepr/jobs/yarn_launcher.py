"""Yarn Launcher Config Interface and Job"""

from dataclasses import dataclass
from typing import Dict, Tuple
import datetime
import json
import logging

from cluster_pack.skein import skein_launcher
import skein

from deepr.config.base import from_config
from deepr.jobs import base
from deepr.jobs.yarn_config import YarnConfig
from deepr.utils import mlflow


LOGGER = logging.getLogger(__name__)


EDITABLE_PACKAGES_INDEX = "editable_packages_index"


@dataclass
class YarnLauncherConfig(YarnConfig):
    """Yarn Launcher Config."""

    name: str = f"yarn-launcher-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    hadoop_file_systems: Tuple[str, ...] = ()
    memory: str = "48 GiB"
    num_cores: int = 48


@dataclass
class YarnLauncher(base.Job):
    """Packages current environment, upload `.pex` and run yarn job."""

    job: Dict
    config: YarnLauncherConfig
    run_on_yarn: bool = True

    def run(self):
        if self.run_on_yarn:
            # Dump job and base as local json files for yarn_launcher
            job_name = f"job-{self.config.name}.json"
            with open(job_name, "w") as file:
                json.dump(self.job, file, indent=4)

            # Launch job on yarn
            pex_path = self.config.upload_pex_cpu()
            with skein.Client() as skein_client:
                LOGGER.info(f"Submitting job {self.config.name}")
                app_id = skein_launcher.submit(
                    skein_client=skein_client,
                    module_name="deepr.cli.main",
                    additional_files=[job_name],
                    package_path=pex_path,
                    args=["from_config", job_name, "-", "run"],
                    env_vars=self.config.get_env_vars(),
                    hadoop_file_systems=list(self.config.hadoop_file_systems),
                    memory=self.config.memory,
                    name=self.config.name,
                    num_cores=self.config.num_cores,
                )
                report = skein_client.application_report(app_id)
                LOGGER.info(f"TRACKING_URL: {report.tracking_url}")
            mlflow.clear_run()
        else:
            LOGGER.info("Not running on yarn.")
            job = from_config(self.job)
            job.run()
