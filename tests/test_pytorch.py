from unittest import TestCase

import numpy as np
from parameterized import parameterized_class

from boost_loss._base import LossBase
from boost_loss._pytorch import _L2LossTorch
from boost_loss.regression import L2Loss

from .test_base import assert_array_almost_equal


@parameterized_class(("loss_torch", "loss_base"), [(_L2LossTorch, L2Loss)])
class TestLossTorch(TestCase):
    loss_torch: type[LossBase]
    loss_base: type[LossBase]

    def setUp(self) -> None:
        self.y_pred = np.random.randn(10)
        self.y_true = np.random.randn(10)

    def test_consistent(self) -> None:
        loss_torch = self.loss_torch()
        loss_base = self.loss_base()
        assert_array_almost_equal(
            loss_torch.loss(self.y_true, self.y_pred),
            loss_base.loss(self.y_true, self.y_pred),
        )
        assert_array_almost_equal(
            loss_torch.grad_hess(self.y_true, self.y_pred)[0],
            loss_base.grad_hess(self.y_true, self.y_pred)[0],
        )
        assert_array_almost_equal(
            loss_torch.grad_hess(self.y_true, self.y_pred)[1],
            loss_base.grad_hess(self.y_true, self.y_pred)[1],
        )
