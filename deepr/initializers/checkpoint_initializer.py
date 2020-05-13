"""Checkpoint Initializer"""

from typing import Dict

import tensorflow as tf


class CheckpointInitializer:
    """Checkpoint Initializer"""

    def __init__(self, path_init_ckpt: str, assignment_map: Dict):
        self.path_init_ckpt = path_init_ckpt
        self.assignment_map = assignment_map

    def __call__(self):
        tf.train.init_from_checkpoint(self.path_init_ckpt, self.assignment_map)
