"""Yarn Launcher Config Interface and Job"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import datetime
import json
import logging
import tempfile

from cluster_pack.skein import skein_config_builder
import getpass
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
                app_id = submit(
                    skein_client=skein_client,
                    module_name="deepr.cli.main",
                    additional_files=[job_name],
                    archive_hdfs=pex_path,
                    args=["from_config", job_name, "-", "run"],
                    env_vars=self.config.get_env_vars(),
                    hadoop_file_systems=self.config.hadoop_file_systems,
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


def submit(
    skein_client: skein.Client,
    module_name: str,
    additional_files: Optional[List[str]] = None,
    archive_hdfs: Optional[str] = None,
    args: Optional[List[str]] = None,
    env_vars: Optional[Dict[str, str]] = None,
    hadoop_file_systems: Tuple[str, ...] = (),
    max_attempts: int = 1,
    max_restarts: int = 0,
    memory: str = "1 GiB",
    name: str = "yarn_launcher",
    node_label: Optional[str] = None,
    num_containers: int = 1,
    num_cores: int = 1,
    pre_script_hook: Optional[str] = None,
    queue: Optional[str] = None,
    user: Optional[str] = None,
) -> str:
    """Submit application via skein."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Update Environment Variables and script hook
        env = dict(env_vars) if env_vars else dict()
        pre_script_hook = pre_script_hook if pre_script_hook else ""
        env.update({"SKEIN_CONFIG": "./.skein", "GIT_PYTHON_REFRESH": "quiet"})

        # Create Skein Config, Service and Spec
        skein_config = skein_config_builder.build(
            module_name,
            args=args if args else [],
            package_path=archive_hdfs,
            additional_files=additional_files,
            tmp_dir=tmp_dir,
        )
        skein_service = skein.Service(
            resources=skein.model.Resources(memory, num_cores),
            instances=num_containers,
            files=skein_config.files,
            env=env,
            script=f"""
                        set -x
                        env
                        {pre_script_hook}
                        {skein_config.script}
                    """,
            max_restarts=max_restarts,
        )
        skein_spec = skein.ApplicationSpec(
            name=name,
            file_systems=list(hadoop_file_systems),
            services={name: skein_service},
            acls=skein.model.ACLs(enable=True, ui_users=["*"], view_users=["*"]),
            max_attempts=max_attempts,
        )

        # Activate impersonation only if user to run the job is not
        # the current user (yarn issue)
        if user and user != getpass.getuser():
            skein_spec.user = user
        if queue:
            skein_spec.queue = queue
        if node_label:
            skein_service.node_label = node_label
        return skein_client.submit(skein_spec)
