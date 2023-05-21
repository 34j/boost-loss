import attrs
import numpy as np
from numpy.typing import NDArray

from ._base import LossBase
from .regression import L1Loss, L2Loss


@attrs.define()
class AsymmetricCompositeLoss(LossBase):
    loss_pred_less: LossBase
    loss_pred_greater: LossBase

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
    def __init__(self, loss: LossBase, t: float = 0.5) -> None:
        super().__init__(loss * (1 - t), loss * t)


class QuantileLoss(AsymmetricLoss):
    def __init__(self, t: float = 0.5) -> None:
        super().__init__(L1Loss(), t)


class ExpectileLoss(AsymmetricLoss):
    def __init__(self, t: float = 0.5) -> None:
        super().__init__(L2Loss(), t)
