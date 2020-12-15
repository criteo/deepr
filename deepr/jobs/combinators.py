"""Pipeline"""

from typing import List
from dataclasses import dataclass
import logging

from deepr.jobs import base


LOGGER = logging.getLogger(__name__)


@dataclass
class Pipeline(base.Job):
    """Pipeline, executes list of jobs in order"""

    jobs: List[base.Job]

    def __post_init__(self):
        for job in self.jobs:
            if not isinstance(job, base.Job):
                raise TypeError(f"Expected `base.Job`, but got {job}")

    def run(self):
        for job in self.jobs:
            job.run()
