from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import attrs

from .base import LossBase

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOG = getLogger(__name__)


@attrs.define()
class DebugLoss(LossBase):
    """Calls LOG.debug() every time loss() or grad_hess() is called."""

    loss_: LossBase

    def loss(self, y_true: NDArray, y_pred: NDArray) -> float | NDArray:
        loss = self.loss_.loss(y_true=y_true, y_pred=y_pred)
        LOG.debug("y_true: %s, y_pred: %s, loss: %s", y_true, y_pred, loss)
        return loss

    def grad_hess(self, y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        grad, hess = self.loss_.grad_hess(y_true=y_true, y_pred=y_pred)
        LOG.debug("y_true: %s, y_pred: %s, grad: %s, hess: %s", y_true, y_pred, grad, hess)
        return grad, hess


@attrs.define()
class PrintLoss(LossBase):
    """Prints every time loss() or grad_hess() is called."""

    loss_: LossBase

    def loss(self, y_true: NDArray, y_pred: NDArray) -> float | NDArray:
        return self.loss_.loss(y_true=y_true, y_pred=y_pred)

    def grad_hess(self, y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        grad, hess = self.loss_.grad_hess(y_true=y_true, y_pred=y_pred)
        return grad, hess
