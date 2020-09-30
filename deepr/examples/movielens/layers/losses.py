# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""Losses."""

import logging
import deepr as dpr

from deepr.examples.movielens.layers.multi import MultiLogLikelihoodCSS
from deepr.examples.movielens.layers.bpr import BPRLoss


LOGGER = logging.getLogger(__name__)


def Loss(loss: str, vocab_size: int):
    """Return the relevant loss layer."""
    if loss == "multi":
        layer = dpr.layers.MultiLogLikelihood(inputs=("logits", "targetPositivesOneHot"), outputs="loss")
    elif loss == "multi_css":
        layer = MultiLogLikelihoodCSS(vocab_size=vocab_size)
    elif loss == "bpr":
        layer = BPRLoss(vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown loss option {loss} (must be 'multi', 'multi_css' or 'bpr')")
    return layer


def VAELoss(loss: str, vocab_size: int, beta_start: float, beta_end: float, beta_steps: int):
    """Add beta * KL to the loss and return relevant loss layer."""
    layer = Loss(loss=loss, vocab_size=vocab_size)
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=tuple(list(layer.inputs) + ["KL"])),
        layer,
        dpr.layers.AddWithWeight(
            inputs=("loss", "KL"), outputs="loss", start=beta_start, end=beta_end, steps=beta_steps
        ),
        dpr.layers.Select(inputs=layer.outputs),
    )
