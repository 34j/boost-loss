from __future__ import annotations

import attrs
import numpy as np
from numpy.typing import NDArray

from ..base import LossBase
from .regression import L1Loss, L2Loss


@attrs.define()
class AsymmetricCompositeLoss(LossBase):
    """Asymmetric composite loss function.
    The loss function is `loss_pred_less` if `y_true < y_pred`,
    otherwise `loss_pred_greater`.
    """

    loss_pred_less: LossBase
    """The loss function if `y_true < y_pred`."""
    loss_pred_greater: LossBase
    """The loss function if `y_true >= y_pred`."""

    def loss(self, y_true: NDArray, y_pred: NDArray) -> float | NDArray:
        loss = np.where(
            y_true < y_pred,
            self.loss_pred_less.loss(y_true=y_true, y_pred=y_pred),
            self.loss_pred_greater.loss(y_true=y_true, y_pred=y_pred),
        )
        return loss

    def grad_hess(self, y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        grad, hess = np.where(
            y_true < y_pred,
            self.loss_pred_less.grad_hess(y_true=y_true, y_pred=y_pred),
            self.loss_pred_greater.grad_hess(y_true=y_true, y_pred=y_pred),
        )
        return grad, hess


class AsymmetricLoss(AsymmetricCompositeLoss):
    """Asymmetric loss function.
    The loss function is `loss * (1 - t)` if `y_true < y_pred`, otherwise `loss * t`.
    Generalized from quantile loss (pinball loss, check loss, etc.) and expectile loss.
    """

    def __init__(self, loss: LossBase, t: float = 0.5) -> None:
        super().__init__(loss * (1 - t), loss * t)


class QuantileLoss(AsymmetricLoss):
    """[Quantile](https://en.wikipedia.org/wiki/Quantile) loss function."""

    def __init__(self, t: float = 0.5) -> None:
        super().__init__(L1Loss(), t)


class ExpectileLoss(AsymmetricLoss):
    r"""[Expectile](https://sites.google.com/site/csphilipps/expectiles) loss function.
    - Expectile is a conditional mean if observations in [μ_τ, ∞)
    are τ/(1 - τ) times more likely than the original distribution.
    - Expectiles are not always quantiles of F.
    - Expectile loss is more smooth than quantile loss.

    .. math::
        \tau \int_{-\infty}^{\mu_\tau} (y - \mu_\tau) dF(y)
        = (1 - \tau) \int_{\mu_\tau}^{\infty} (y - \mu_\tau) dF(y)"""

    def __init__(self, t: float = 0.5) -> None:
        super().__init__(L2Loss(), t)
