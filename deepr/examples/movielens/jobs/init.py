"""Init Checkpoint with SVD embeddings."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

import deepr


LOGGER = logging.getLogger(__name__)


@dataclass
class InitCheckpoint(deepr.jobs.Job):
    """Init Checkpoint with SVD embeddings."""

    path_embeddings: str
    path_init_ckpt: str
    normalize: bool
    path_counts: Optional[str] = None
    use_log_counts: bool = True
    normalize_counts: bool = True

    def run(self):
        # Reload NumPy embeddings
        LOGGER.info(f"Reloading embeddings from {self.path_embeddings}")
        with deepr.io.Path(self.path_embeddings).open("rb") as file:
            embeddings = np.load(file)
            embeddings = embeddings.astype(np.float32)

        # Normalize and create variables dictionary
        if self.normalize:
            LOGGER.info("Normalizing product embeddings.")
            embeddings = np.divide(embeddings, np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        variables = {"embeddings": embeddings}

        # Reload counts, transform with log, add to variables
        if self.path_counts is not None:
            LOGGER.info(f"Reloading counts from {self.path_counts}")
            with deepr.io.Path(self.path_counts).open("rb") as file:
                counts = np.load(file)
                counts = counts.astype(np.float32)
                if self.normalize_counts:
                    counts /= np.sum(counts)
                if self.use_log_counts:
                    counts = np.log(counts)
                variables["biases"] = counts

        # Save variables dictionary in checkpoint
        LOGGER.info(f"Saving embeddings in checkpoint {self.path_init_ckpt}")
        deepr.io.Path(self.path_init_ckpt).parent.mkdir(parents=True, exist_ok=True)
        deepr.utils.save_variables_in_ckpt(path=self.path_init_ckpt, variables=variables)
