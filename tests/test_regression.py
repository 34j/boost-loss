from unittest import TestCase

import numpy as np

from boost_loss.regression import L1Loss, L2Loss


class TestRegression(TestCase):
    def test_l2(self):
        l2 = L2Loss()
        y_pred = np.random.rand(10)
        y_true = np.random.rand(10)
        loss = l2.loss(y_true=y_true, y_pred=y_pred)
        grad = l2.grad(y_true=y_true, y_pred=y_pred)
        hess = l2.hess(y_true=y_true, y_pred=y_pred)
        self.assertEqual(grad.tolist(), (y_pred - y_true).tolist())
        self.assertEqual(hess.tolist(), np.ones_like(y_pred).tolist())
        self.assertEqual(loss.tolist(), ((y_pred - y_true) ** 2).tolist())

    def test_l1(self):
        l1 = L1Loss()
        y_pred = np.random.rand(10)
        y_true = np.random.rand(10)
        loss = l1.loss(y_true=y_true, y_pred=y_pred)
        grad = l1.grad(y_true=y_true, y_pred=y_pred)
        hess = l1.hess(y_true=y_true, y_pred=y_pred)
        self.assertEqual(grad.tolist(), np.sign(y_pred - y_true).tolist())
        self.assertEqual(hess.tolist(), np.zeros_like(y_pred).tolist())
        self.assertEqual(loss.tolist(), np.abs(y_pred - y_true).tolist())
