"""Copy Directory"""

from dataclasses import dataclass
import logging

from deepr.jobs import base
from deepr.io.path import Path


LOGGER = logging.getLogger(__name__)


@dataclass
class CopyDir(base.Job):
    """Copy Directory"""

    source: str
    target: str

    def run(self):
        LOGGER.info(f"Copying {self.source} to {self.target}")
        Path(self.source).copy_dir(self.target)
