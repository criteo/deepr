"""Graphite Utilities"""

import logging
import os

import graphyte

from deepr.utils.exceptions import handle_exceptions


LOGGER = logging.getLogger(__name__)


HOST_ENV_VAR = "GRAPHITE_HOST"

PORT_ENV_VAR = "GRAPHITE_PORT"

PREFIX_ENV_VAR = "GRAPHITE_PREFIX"

INTERVAL_ENV_VAR = "GRAPHITE_INTERVAL"


@handle_exceptions
def get_sender(
    host: str = None,
    port: int = None,
    prefix: str = None,
    postfix: str = None,
    timeout: int = 5,
    interval: int = None,
    queue_size: int = None,
    log_sends: bool = False,
    protocol: str = "tcp",
    batch_size: int = 1000,
):
    """Get Graphite Sender."""
    # Retrieve parameters from environment variables
    host = host if host else os.environ[HOST_ENV_VAR]
    port = port if port else int(os.environ[PORT_ENV_VAR])
    prefix = prefix if prefix else os.environ[PREFIX_ENV_VAR]
    interval = interval if interval else int(os.environ[INTERVAL_ENV_VAR])

    # Return sender
    prefix = f"{prefix}.{postfix}" if postfix else prefix
    return graphyte.Sender(
        host=host,
        port=port,
        prefix=prefix,
        timeout=timeout,
        interval=interval,
        queue_size=queue_size,
        log_sends=log_sends,
        protocol=protocol,
        batch_size=batch_size,
    )


@handle_exceptions
def log_metric(metric, value, postfix: str = None):
    """Log metric to graphite."""
    sender = get_sender(postfix=postfix)
    sender.send(metric, _safe_value(value))


@handle_exceptions
def log_metrics(metrics, postfix: str = None):
    """Log metrics to graphite"""
    sender = get_sender(postfix=postfix)
    for metric, value in metrics.items():
        sender.send(metric, _safe_value(value))


def _safe_value(value):
    """Converts value to int or float"""
    if isinstance(value, (float, int)):
        return value
    value_int = int(value)
    value_float = float(value)
    return value_int if value_int == value_float else value_float
