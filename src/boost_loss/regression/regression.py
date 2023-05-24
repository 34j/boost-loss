from logging import getLogger

import attrs
import numpy as np
from numpy.typing import NDArray

from .._base import LossBase

LOG = getLogger(__name__)


# cannot freeze due to FrozenInstanceError in catboost
@attrs.define()
class LNLoss(LossBase):
    """LNLoss = |y_true - y_pred|^n
    - x < 1 is not recommended because the loss is not convex.
    - x >> 2 is not recommended because the gradient is too steep."""

    n: float
    """The exponent of the loss."""
    divide_n_loss: bool = False
    """Whether to divide the loss by n. Generally False is used."""
    divide_n_grad: bool = True
    """Whether to divide the gradient by n. Generally True is used."""

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
    """L1 loss = |y_true - y_pred|."""

    def __init__(
        self, *, divide_n_loss: bool = False, divide_n_grad: bool = True
    ) -> None:
        """L1 loss.

        Parameters
        ----------
        divide_n_loss : bool, optional
            Whether to divide the loss by n, by default False
        divide_n_grad : bool, optional
            Whether to divide the gradient by n, by default True
        """
        super().__init__(n=1, divide_n_loss=divide_n_loss, divide_n_grad=divide_n_grad)


class L2Loss(LNLoss):
    """L2 loss = |y_true - y_pred|^2"""

    def __init__(
        self, *, divide_n_loss: bool = False, divide_n_grad: bool = True
    ) -> None:
        """L2 loss.

        Parameters
        ----------
        divide_n_loss : bool, optional
            Whether to divide the loss by n, by default False
        divide_n_grad : bool, optional
            Whether to divide the gradient by n, by default True
        """
        super().__init__(n=2, divide_n_loss=divide_n_loss, divide_n_grad=divide_n_grad)
