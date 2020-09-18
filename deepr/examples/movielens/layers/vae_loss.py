# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""Multinomial Loss for the Multi-VAE."""

import logging
import deepr as dpr


LOGGER = logging.getLogger(__name__)


def VAELoss(beta_start: float, beta_end: float, beta_steps: int):
    """Compute Multi-VAE loss."""
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("logits", "inputPositivesOneHot", "KL")),
        dpr.layers.MultiLogLikelihood(inputs=("logits", "inputPositivesOneHot"), outputs="loss_multi"),
        dpr.layers.AddWithWeight(
            inputs=("loss_multi", "KL"), outputs="loss", start=beta_start, end=beta_end, steps=beta_steps
        ),
        dpr.layers.Select(inputs=("loss", "loss_multi")),
    )
