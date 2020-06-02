"""Copy Directory"""

from dataclasses import dataclass
import logging

from deepr.jobs import base
from deepr.io.path import Path


LOGGER = logging.getLogger(__name__)


@dataclass
class CopyDir(base.Job):
    """Copy Directory. Overwrite destination by default"""

    source: str
    target: str
    skip_copy: bool = False
    overwrite: bool = True

    def run(self):
        if self.skip_copy:
            LOGGER.info(f"NOT COPYING {self.source} to {self.target} (skip_copy=True)")
            return
        if self.overwrite_destination_folder and Path(self.target).is_dir():
            Path(self.target).delete_dir()

        LOGGER.info(f"Copying {self.source} to {self.target}")
        Path(self.source).copy_dir(self.target)
