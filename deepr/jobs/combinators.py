"""Pipeline"""

from typing import List, Union, Callable
from dataclasses import dataclass
import logging

from deepr.jobs import base


LOGGER = logging.getLogger(__name__)


@dataclass
class Pipeline(base.Job):
    """Pipeline, executes list of jobs in order"""

    jobs: List[Union[base.Job, Callable]]

    def __post_init__(self):
        for job in self.jobs:
            if not (hasattr(job, "run") or callable(job)):
                raise TypeError(f"Expected `base.Job` or function, but got {job}")

    def run(self):
        for job in self.jobs:
            if callable(job):
                job()
            else:
                job.run()
