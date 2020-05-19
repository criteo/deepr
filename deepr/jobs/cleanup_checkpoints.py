"""Cleanup Checkpoints in path_model"""

from dataclasses import dataclass
import logging

from deepr.jobs import base
from deepr.io.path import Path


LOGGER = logging.getLogger(__name__)


@dataclass
class CleanupCheckpoints(base.Job):
    """Cleanup Checkpoints in path_model"""

    path_model: str
    path_checkpoints: str = "checkpoints"

    def run(self):
        LOGGER.info(f"Cleanup checkpoints in {self.path_model}/{self.path_checkpoints}")
        checkpoint_files = Path(self.path_model, self.path_checkpoints).glob("model.ckpt-*")
        for checkpoint_file in checkpoint_files:
            LOGGER.info(f"- Deleting {checkpoint_file}")
            checkpoint_file.delete()
