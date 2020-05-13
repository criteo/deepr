"""Interface for jobs"""

from abc import ABC, abstractmethod
import logging


LOGGER = logging.getLogger(__name__)


class Job(ABC):
    """Interface for jobs"""

    @abstractmethod
    def run(self):
        """Run Job"""
        raise NotImplementedError()
