"""Initializes Graphite Variables."""

import logging
import os

from deepr.utils.graphite import HOST_ENV_VAR, PORT_ENV_VAR, PREFIX_ENV_VAR, INTERVAL_ENV_VAR


LOGGER = logging.getLogger(__name__)


class GraphiteInit(dict):
    """Initializes Graphite Variables."""

    def __init__(
        self, host: str = None, prefix: str = None, port: int = 3341, interval: int = 60, use_graphite: bool = False
    ):
        if use_graphite:
            # Check arguments are not None
            if host is None:
                raise ValueError("'host' should not be None")
            if prefix is None:
                raise ValueError("'prefix' should not be None")

            # Store variables in environment variables
            os.environ[HOST_ENV_VAR] = host
            os.environ[PORT_ENV_VAR] = str(port)
            os.environ[PREFIX_ENV_VAR] = prefix
            os.environ[INTERVAL_ENV_VAR] = str(interval)

            # Log variables
            LOGGER.info(f"{HOST_ENV_VAR}: {host}")
            LOGGER.info(f"{PORT_ENV_VAR}: {port}")
            LOGGER.info(f"{PREFIX_ENV_VAR}: {prefix}")
            LOGGER.info(f"{INTERVAL_ENV_VAR}: {interval}")
        else:
            LOGGER.warning("Graphite not initialized.")

        # Store in dictionary
        super().__init__(use_graphite=use_graphite, host=host, prefix=prefix, port=port, interval=interval)
