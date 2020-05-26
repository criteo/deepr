"""Basic Yarn Config in charge of uploading environment."""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import logging
import os

from tf_yarn import packaging
import cluster_pack

from deepr.io.path import Path
from deepr.utils.graphite import HOST_ENV_VAR, PORT_ENV_VAR, PREFIX_ENV_VAR, INTERVAL_ENV_VAR
from deepr.utils.mlflow import RUN_ID_ENV_VAR, TRACKING_URI_ENV_VAR


LOGGER = logging.getLogger(__name__)


@dataclass
class YarnConfig:
    """Basic Yarn Config in charge of uploading environment."""

    name: str
    gpu_additional_packages: Tuple[str, ...] = ("tensorflow-gpu==1.15.2", "tf-yarn-gpu==0.4.19")
    gpu_ignored_packages: Tuple[str, ...] = ("tensorflow", "tf-yarn")
    gpu_to_use: Optional[int] = None
    jvm_memory_in_gb: int = 8
    path_pex_cpu: Optional[str] = None
    path_pex_gpu: Optional[str] = None
    path_pex_prefix: str = f"viewfs://root/user/{os.environ.get('USER', 'deepr')}/envs"

    def get_env_vars(self):
        """Return Default Environment Variables"""
        # JVM environment Variables
        env_vars = {
            "LIBHDFS_OPTS": f"-Xms{self.jvm_memory_in_gb}g -Xmx{self.jvm_memory_in_gb}g",
            "MALLOC_ARENA_MAX": "0",
        }
        # GPU Environment Variables
        if self.gpu_to_use is not None:
            env_vars["CUDA_VISIBLE_DEVICES"] = str(self.gpu_to_use)
        # MLFlow Environment Variables
        if os.environ.get(RUN_ID_ENV_VAR) is not None:
            env_vars[RUN_ID_ENV_VAR] = os.environ.get(RUN_ID_ENV_VAR)
            env_vars[TRACKING_URI_ENV_VAR] = os.environ.get(TRACKING_URI_ENV_VAR)
        # Graphite Environment Variables
        if os.environ.get(PREFIX_ENV_VAR) is not None:
            env_vars[HOST_ENV_VAR] = os.environ.get(HOST_ENV_VAR)
            env_vars[PORT_ENV_VAR] = os.environ.get(PORT_ENV_VAR)
            env_vars[PREFIX_ENV_VAR] = os.environ.get(PREFIX_ENV_VAR)
            env_vars[INTERVAL_ENV_VAR] = os.environ.get(INTERVAL_ENV_VAR)
        return env_vars

    def upload_pex_cpu(self) -> str:
        """Upload Current Environment as PEX for CPU."""
        path_pex = f"{self.path_pex_prefix}/cpu/{self.name}.pex"
        return upload_pex(path_pex=path_pex, path_pex_existing=self.path_pex_cpu)

    def upload_pex_gpu(self) -> str:
        """Upload Current Environment as PEX for GPU."""
        path_pex = f"{self.path_pex_prefix}/gpu/{self.name}.pex"
        return upload_pex(
            path_pex=path_pex,
            path_pex_existing=self.path_pex_gpu,
            additional_packages=dict((req.split("==")[0], req.split("==")[1]) for req in self.gpu_additional_packages),
            ignored_packages=list(self.gpu_ignored_packages) if self.gpu_ignored_packages else None,
        )


def upload_pex(
    path_pex: str, path_pex_existing: str = None, additional_packages: Dict = None, ignored_packages: List = None
) -> str:
    """Upload Current Environment and return path to PEX on HDFS"""
    if path_pex_existing is None:
        LOGGER.info(f"Uploading env to {path_pex}")
        packaging.upload_env_to_hdfs(
            archive_on_hdfs=path_pex,
            additional_packages=additional_packages if additional_packages else {},
            ignored_packages=ignored_packages if ignored_packages else [],
            packer=cluster_pack.packaging.PEX_PACKER,
        )
    elif not Path(path_pex_existing).is_hdfs:
        LOGGER.info(f"Uploading env to {path_pex}")
        packaging.upload_zip_to_hdfs(path_pex_existing, archive_on_hdfs=path_pex)
    else:
        LOGGER.info(f"Skipping upload, PEX {path_pex_existing} already exists")
        path_pex = path_pex_existing
    return path_pex
