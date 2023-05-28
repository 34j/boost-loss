from __future__ import annotations

from logging import getLogger

import attrs
import numpy as np
from numpy.typing import NDArray

from ..base import LossBase

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

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return np.ones_like(y_pred - y_true)  # zero hess is not allowed


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


class LogCoshLoss(LossBase):
    """LogCosh loss = log(cosh(y_true - y_pred))"""

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return np.log(np.cosh(y_pred - y_true))

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return np.tanh(y_pred - y_true)

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return np.cosh(y_pred - y_true) ** -2


class HuberLoss(LossBase):
    """Huber loss = 0.5 (y_true - y_pred)^2 if |y_true - y_pred| <= delta
    else delta * (|y_true - y_pred| - 0.5 * delta)"""

    def __init__(self, delta: float = 1.0) -> None:
        """Huber loss.

        Parameters
        ----------
        delta : float, optional
            The parameter delta, by default 1.0
        """
        self.delta = delta

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        y_diff = y_pred - y_true
        return np.where(
            np.abs(y_diff) <= self.delta,
            0.5 * y_diff**2,
            self.delta * (np.abs(y_diff) - 0.5 * self.delta),
        )

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        y_diff = y_pred - y_true
        return np.where(
            np.abs(y_diff) <= self.delta,
            y_diff,
            np.sign(y_diff) * self.delta,
        )

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return np.ones_like(y_pred - y_true)  # zero hess is not allowed
        # return np.where(
        #     np.abs(y_pred - y_true) <= self.delta,
        #     np.ones_like(y_true),
        #     np.zeros_like(y_true),
        # )


class FairLoss(LossBase):
    """Fair loss = c^2/2 * (abs(y_true - y_pred) -
    c * log(1 + abs(y_true - y_pred)/c))"""

    def __init__(self, c: float = 1.0) -> None:
        """Fair loss.

        Parameters
        ----------
        c : float, optional
            The parameter c, by default 1.0
        """
        self.c = c

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return (
            self.c**2
            / 2
            * (
                np.abs(y_true - y_pred)
                - self.c * np.log(1 + np.abs(y_true - y_pred) / self.c)
            )
        )

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return (
            self.c**2
            / 2
            * (np.sign(y_pred - y_true) - self.c / (self.c + np.abs(y_pred - y_true)))
        )

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return (
            self.c**2
            / 2
            * (np.zeros_like(y_true) + self.c / (self.c + np.abs(y_pred - y_true)) ** 2)
        )


class PoissonLoss(LossBase):
    """Poisson loss = y_pred - y_true * log(y_pred)"""

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return y_pred - y_true * np.log(y_pred)

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return 1 - y_true / y_pred

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return y_true / y_pred**2


class LogLoss(LossBase):
    """Log loss = log(1 + exp(-y_true * y_pred))"""

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return np.log(1 + np.exp(-y_true * y_pred))

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return -y_true / (1 + np.exp(y_true * y_pred))

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return (
            y_true**2 * np.exp(y_true * y_pred) / (1 + np.exp(y_true * y_pred)) ** 2
        )


class MSLELoss(LossBase):
    """Mean squared logarithmic error loss = (log(1 + y_true) - log(1 + y_pred))^2"""

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return np.square(np.log1p(y_true) - np.log1p(y_pred))

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return -2 * (np.log1p(y_true) - np.log1p(y_pred)) / (1 + y_pred)

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return 2 * (np.log1p(y_true) - np.log1p(y_pred)) / (1 + y_pred) ** 2


class MAPELoss(LossBase):
    """Mean absolute percentage error loss = abs(y_true - y_pred) / y_true"""

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return np.abs(y_true - y_pred) / y_true

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return np.sign(y_pred - y_true) / y_true

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return np.zeros_like(y_true)


class SMAPELoss(LossBase):
    """Symmetric mean absolute percentage error loss =
    abs(y_true - y_pred) / ((abs(y_true) + abs(y_pred))/2)"""

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return np.sign(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return np.zeros_like(y_true)


class TweedieLoss(LossBase):
    """Tweedie loss = -y_true * y_pred^(2-p) / (2-p) + y_pred^(1-p) / (1-p)"""

    def __init__(self, p: float = 1.5) -> None:
        """Tweedie loss.

        Parameters
        ----------
        p : float, optional
            The parameter p, by default 1.5
        """
        self.p = p

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return -y_true * y_pred ** (2 - self.p) / (2 - self.p) + y_pred ** (
            1 - self.p
        ) / (1 - self.p)

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return -y_true * (2 - self.p) * y_pred ** (1 - self.p) + y_pred ** (-self.p)

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return y_true * (2 - self.p) * (1 - self.p) * y_pred ** (
            -self.p - 1
        ) - self.p * y_pred ** (-self.p - 1)


class GammaLoss(LossBase):
    """Gamma loss = y_true / y_pred - log(y_true / y_pred) - 1"""

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return y_true / y_pred - np.log(y_true / y_pred) - 1

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return -y_true / y_pred**2 + 1 / y_pred

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return 2 * y_true / y_pred**3 - 1 / y_pred**2
