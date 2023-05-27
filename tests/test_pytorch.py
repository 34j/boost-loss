from unittest import TestCase

import numpy as np
from parameterized import parameterized_class
from torch.nn.modules.loss import MSELoss

from boost_loss.base import LossBase
from boost_loss.regression.regression import L2Loss
from boost_loss.torch import TorchLossBase, _LNLossTorch, _LNLossTorch_

from .test_base import assert_array_almost_equal


@parameterized_class(
    ("loss_torch", "loss_base"),
    [
        (_LNLossTorch_(n=2, divide_n_loss=False), L2Loss(divide_n_grad=False)),
        (_LNLossTorch(n=2, divide_n_grad=True), L2Loss(divide_n_grad=True)),
        (TorchLossBase.from_callable_torch(MSELoss())(), L2Loss(divide_n_grad=False)),
    ],
)
class TestLossTorch(TestCase):
    loss_torch: TorchLossBase
    loss_base: LossBase

    def setUp(self) -> None:
        self.y_pred = np.random.randn(10)
        self.y_true = np.random.randn(10)

    def test_consistent(self) -> None:
        loss_torch = self.loss_torch
        loss_base = self.loss_base
        assert_array_almost_equal(
            np.mean(loss_torch.loss(self.y_true, self.y_pred)),
            np.mean(loss_base.loss(self.y_true, self.y_pred)),
        )
        assert_array_almost_equal(
            loss_torch.grad_hess(self.y_true, self.y_pred)[0],
            loss_base.grad_hess(self.y_true, self.y_pred)[0],
        )
        assert_array_almost_equal(
            loss_torch.grad_hess(self.y_true, self.y_pred)[1],
            loss_base.grad_hess(self.y_true, self.y_pred)[1],
        )
