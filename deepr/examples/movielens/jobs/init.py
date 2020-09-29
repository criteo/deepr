"""Init Checkpoint with SVD embeddings."""

import logging
from dataclasses import dataclass

import numpy as np

import deepr as dpr


LOGGER = logging.getLogger(__name__)


@dataclass
class InitCheckpoint(dpr.jobs.Job):
    """Init Checkpoint with SVD embeddings."""

    path_embeddings: str
    path_init_ckpt: str
    normalize: bool

    def run(self):
        LOGGER.info(f"Reloading embeddings from {self.path_embeddings}")
        with dpr.io.Path(self.path_embeddings).open("rb") as file:
            embeddings = np.load(file)
            embeddings = embeddings.astype(np.float32)

        if self.normalize:
            LOGGER.info("Normalizing product embeddings.")
            embeddings = np.divide(embeddings, np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        LOGGER.info(f"Saving embeddings in checkpoint {self.path_init_ckpt}")
        dpr.io.Path(self.path_init_ckpt).parent.mkdir(parents=True, exist_ok=True)
        dpr.utils.save_variables_in_ckpt(path=self.path_init_ckpt, variables={"embeddings": embeddings})
