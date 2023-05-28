from __future__ import annotations

from logging import getLogger

import attrs
from numpy.typing import NDArray

from .base import LossBase

LOG = getLogger(__name__)


@attrs.define()
class DebugLoss(LossBase):
    """Calls LOG.debug() every time loss() or grad_hess() is called."""

    loss_: LossBase

    def loss(self, y_true: NDArray, y_pred: NDArray) -> float | NDArray:
        loss = self.loss_.loss(y_true=y_true, y_pred=y_pred)
        LOG.debug(f"y_true: {y_true}, y_pred: {y_pred}, loss: {loss}")
        return loss

    def grad_hess(self, y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        grad, hess = self.loss_.grad_hess(y_true=y_true, y_pred=y_pred)
        LOG.debug(f"y_true: {y_true}, y_pred: {y_pred}, grad: {grad}, hess: {hess}")
        return grad, hess


@attrs.define()
class PrintLoss(LossBase):
    """Prints every time loss() or grad_hess() is called."""

    loss_: LossBase

    def loss(self, y_true: NDArray, y_pred: NDArray) -> float | NDArray:
        loss = self.loss_.loss(y_true=y_true, y_pred=y_pred)
        print(f"y_true: {y_true}, y_pred: {y_pred}, loss: {loss}")
        return loss

    def grad_hess(self, y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        grad, hess = self.loss_.grad_hess(y_true=y_true, y_pred=y_pred)
        print(f"y_true: {y_true}, y_pred: {y_pred}, grad: {grad}, hess: {hess}")
        return grad, hess
