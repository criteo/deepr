"""Summary Saver Hook"""

from typing import List, Dict

import tensorflow as tf

from deepr.hooks.base import TensorHookFactory


class SummarySaverHookFactory(TensorHookFactory):
    """Summary Saver Hook"""

    def __init__(
        self,
        tensors: List[str] = None,
        save_steps: int = None,
        save_secs: int = None,
        output_dir: str = None,
        summary_writer=None,
        scaffold=None,
    ):
        self.tensors = tensors
        self.save_steps = save_steps
        self.save_secs = save_secs
        self.output_dir = output_dir
        self.summary_writer = summary_writer
        self.scaffold = scaffold

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> tf.estimator.SessionRunHook:
        # Extract relevant tensors
        if self.tensors is None:
            tensors = {key: tensor for key, tensor in tensors.items() if len(tensor.shape) == 0}
        else:
            tensors = {name: tensors[name] for name in self.tensors}

        # Define summaries
        for name, tensor in tensors.items():
            tf.summary.scalar(name, tensor)

        return tf.train.SummarySaverHook(
            save_steps=self.save_steps,
            save_secs=self.save_secs,
            output_dir=self.output_dir,
            summary_writer=self.summary_writer,
            scaffold=self.scaffold,
            summary_op=tf.summary.merge_all(),
        )
