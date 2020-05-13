"""Yarn Launcher Config Interface and Job"""

from abc import ABC
import atexit
import json
from dataclasses import dataclass
import logging
import tempfile
from typing import Dict, Optional, List

import skein
from mlflow import end_run
import getpass
from cluster_pack.skein import skein_config_builder

from deepr.jobs import base
from deepr.config.base import from_config


EDITABLE_PACKAGES_INDEX = "editable_packages_index"


LOGGER = logging.getLogger(__name__)


@dataclass
class YarnLauncherConfig(ABC):
    """Abstract Yarn Launcher Config."""

    name: str = "default-name"
    num_cores: int = 48
    memory: str = "48 GiB"

    @property
    def env_vars(self) -> Dict[str, str]:
        """Create environment variables from config"""
        raise NotImplementedError()

    def upload_env(self) -> str:
        """Upload environment to HDFS"""
        raise NotImplementedError()


@dataclass
class YarnLauncher(base.Job):
    """Packages current environment, upload `.pex` and run yarn job."""

    job: Dict
    config: YarnLauncherConfig

    def __post_init__(self):
        job = from_config(self.job)
        if not isinstance(job, base.Job):
            raise TypeError(f"Expected type {base.Job} but got {job}")

    def run(self):
        # Dump job and base as local json files for yarn_launcher
        job_name = f"job-{self.config.name}.json"
        with open(job_name, "w") as file:
            json.dump(self.job, file, indent=4)

        # Remove auto-termination of active MLFlow runs because they
        # are forwarded on yarn
        atexit.unregister(end_run)

        # Launch job on yarn
        pex_path = self.config.upload_env()
        with skein.Client() as skein_client:
            LOGGER.info(f"Submitting job {self.config.name}")
            app_id = submit(
                skein_client=skein_client,
                module_name="deepr.cli.main",
                archive_hdfs=pex_path,
                additional_files=[job_name],
                args=["from_config_file", job_name, "-", "run"],
                name=self.config.name,
                num_cores=self.config.num_cores,
                memory=self.config.memory,
                env_vars=self.config.env_vars,
            )
            report = skein_client.application_report(app_id)
            LOGGER.info(f"TRACKING_URL: {report.tracking_url}")


def submit(
    skein_client: skein.Client,
    module_name: str,
    args: Optional[List[str]] = None,
    name: str = "yarn_launcher",
    num_cores: int = 1,
    memory: str = "1 GiB",
    archive_hdfs: Optional[str] = None,
    hadoop_file_systems: Optional[List[str]] = None,
    queue: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
    additional_files: Optional[List[str]] = None,
    node_label: Optional[str] = None,
    num_containers: int = 1,
    user: Optional[str] = None,
    acquire_map_reduce_delegation_token: bool = False,
    pre_script_hook: Optional[str] = None,
    max_attempts: int = 1,
    max_restarts: int = 0,
) -> str:
    """submit"""
    if hadoop_file_systems is None:
        hadoop_file_systems = ["viewfs://root", "viewfs://prod-am6", "viewfs://prod-pa4", "viewfs://preprod-pa4"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        skein_config = skein_config_builder.build(
            module_name,
            args=args if args else [],
            package_path=archive_hdfs,
            additional_files=additional_files,
            tmp_dir=tmp_dir,
        )

        env = dict(env_vars) if env_vars else dict()
        pre_script_hook = pre_script_hook if pre_script_hook else ""
        env.update({"SKEIN_CONFIG": "./.skein", "GIT_PYTHON_REFRESH": "quiet"})

        service = skein.Service(
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

        spec = skein.ApplicationSpec(
            name=name,
            file_systems=hadoop_file_systems,
            services={name: service},
            acls=skein.model.ACLs(enable=True, ui_users=["*"], view_users=["*"]),
            acquire_map_reduce_delegation_token=acquire_map_reduce_delegation_token,
            max_attempts=max_attempts,
        )

        # activate impersonification only if user to run the job is not the current user (yarn issue)
        if user and user != getpass.getuser():
            spec.user = user

        if queue:
            spec.queue = queue

        if node_label:
            service.node_label = node_label

        return skein_client.submit(spec)
