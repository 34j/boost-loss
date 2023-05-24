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
    divide_n_loss: bool = False
    divide_n_grad: bool = True

    @property
    def name(self) -> str:
        return f"l{self.n:.2f}"

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return np.abs(y_pred - y_true) ** self.n / (self.n if self.divide_n_loss else 1)

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        y_diff = y_pred - y_true
        return (
            np.abs(y_diff) ** (self.n - 1)
            * np.sign(y_diff)
            * self.n
            / (self.n if self.divide_n_grad else 1)
        )

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        y_diff = y_pred - y_true
        return (
            (self.n - 1)
            * np.abs(y_diff) ** (self.n - 2)
            * self.n
            / (self.n if self.divide_n_grad else 1)
        )


class L1Loss(LNLoss):
    def __init__(
        self, *, divide_n_loss: bool = False, divide_n_grad: bool = True
    ) -> None:
        super().__init__(n=1, divide_n_loss=divide_n_loss, divide_n_grad=divide_n_grad)


class L2Loss(LNLoss):
    def __init__(
        self, *, divide_n_loss: bool = False, divide_n_grad: bool = True
    ) -> None:
        super().__init__(n=2, divide_n_loss=divide_n_loss, divide_n_grad=divide_n_grad)
