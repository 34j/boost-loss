from logging import getLogger

import attrs
import numpy as np
from numpy.typing import NDArray

from ._base import LossBase

LOG = getLogger(__name__)


# cannot freeze due to FrozenInstanceError in catboost
@attrs.define()
class LNLoss(LossBase):
    """LNLoss = |y_true - y_pred|^x
    x < 1 is not recommended because the loss is not convex
    (and not differentiable at y_true = y_pred).
    x >> 2 is not recommended because the gradient is too steep."""

    n: float

    @property
    def name(self) -> str:
        return f"l{self.n:.2f}"

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return np.abs(y_pred - y_true) ** self.n

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        y_diff = y_pred - y_true
        return self.n * np.abs(y_diff) ** (self.n - 1) * np.sign(y_diff)

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        y_diff = y_pred - y_true
        return self.n * (self.n - 1) * np.abs(y_diff) ** (self.n - 2)


class L1Loss(LNLoss):
    def __init__(self) -> None:
        super().__init__(n=1)


class L2Loss(LNLoss):
    def __init__(self) -> None:
        super().__init__(n=2)


@attrs.define
class DebugLoss(LossBase):
    loss_: LossBase

    def loss(self, y_true: NDArray, y_pred: NDArray) -> float | NDArray:
        loss = self.loss_.loss(y_true=y_true, y_pred=y_pred)
        LOG.debug(f"y_pred: {y_pred}, loss: {loss}")
        return loss

    def grad_hess(self, y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        grad, hess = self.loss_.grad_hess(y_true=y_true, y_pred=y_pred)
        LOG.debug(f"y_pred: {y_pred}, grad: {grad}, hess: {hess}")
        return grad, hess
