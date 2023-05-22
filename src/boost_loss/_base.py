from __future__ import annotations

import warnings
from abc import ABCMeta
from logging import getLogger
from numbers import Real
from typing import Any, Callable, Sequence, final

import attrs
import humps
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from numpy.typing import NDArray
from typing_extensions import Self

LOG = getLogger(__name__)


def _dataset_to_ndarray(
    y: NDArray | lgb.Dataset | xgb.DMatrix,
) -> tuple[NDArray, NDArray]:
    if isinstance(y, lgb.Dataset):
        y_ = y.get_label()
        if y_ is None:
            raise ValueError("y is None")
        weight = y.get_weight()
        if weight is None:
            weight = np.ones_like(y_)
        return y_, weight
    if isinstance(y, xgb.DMatrix):
        y_ = y.get_label()
        return y_, np.ones_like(y_)
    return y, np.ones_like(y)


class LossBase(metaclass=ABCMeta):
    """Base class for loss functions.

    See Also
    --------
    Catboost:
    https://catboost.ai/en/docs/concepts/python-usages-examples#user-defined-loss-function
    LightGBM:
    https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#custom-objective-function
    XGBoost:
    https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html
    """

    is_higher_better: bool = False

    @classmethod
    def from_function(
        cls,
        name: str,
        loss: Callable[[NDArray, NDArray], NDArray],
        grad: Callable[[NDArray, NDArray], NDArray],
        hess: Callable[[NDArray, NDArray], NDArray],
        is_higher_better: bool = False,
    ) -> type[LossBase]:
        return attrs.make_class(
            name,
            bases=(cls,),
            attrs=dict(
                loss=loss,
                grad=grad,
                hess=hess,
                is_higher_better=is_higher_better,
            ),
        )

    @property
    def grad_hess_sign(self) -> int:
        return -1 if self.is_higher_better else 1

    def __init_subclass__(cls, **kwargs: Any) -> None:
        grad_inherited = cls.grad is not LossBase.grad
        hess_inherited = cls.hess is not LossBase.hess
        grad_hess_inherited = cls.grad_hess is not LossBase.grad_hess
        if grad_inherited and hess_inherited:
            pass
        elif grad_hess_inherited:
            pass
        else:
            raise TypeError(
                f"Can't instantiate abstract class {cls.__name__} "
                "with grad_hess or both grad and hess not implemented"
            )
        super().__init_subclass__(**kwargs)

    @property
    def name(self) -> str:
        return humps.camelize(self.__class__.__name__)

    @property
    def __name__(self) -> str:
        return self.name

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        raise NotImplementedError()

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        raise NotImplementedError()

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray | float:
        raise NotImplementedError()

    def grad_hess(self, y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        return self.grad(y_true=y_true, y_pred=y_pred), self.hess(
            y_true=y_true, y_pred=y_pred
        )

    @final
    def __call__(
        self,
        y_true: NDArray | lgb.Dataset | xgb.DMatrix,
        y_pred: NDArray | lgb.Dataset | xgb.DMatrix,
    ) -> tuple[NDArray, NDArray]:
        """Sklearn-compatible interface (Sklearn, LightGBM, XGBoost)"""
        if isinstance(y_pred, lgb.Dataset) or isinstance(y_pred, xgb.DMatrix):
            # NOTE: swap (it is so fucking that the order is inconsistent)
            y_true, y_pred = y_pred, y_true
        y_true, weight = _dataset_to_ndarray(y=y_true)
        y_pred, _ = _dataset_to_ndarray(y=y_pred)
        grad, hess = self.grad_hess(y_true=y_true, y_pred=y_pred)
        grad, hess = grad * weight, hess * weight
        grad, hess = grad * self.grad_hess_sign, hess * self.grad_hess_sign
        return grad, hess

    @final
    def calc_ders_range(
        self,
        preds: Sequence[float],
        targets: Sequence[float],
        weights: Sequence[float] | None = None,
    ) -> list[tuple[float, float]]:
        """Catboost-compatible interface"""
        preds_ = np.array(preds)
        targets_ = np.array(targets)
        weights_ = np.array(weights) if weights is not None else np.ones_like(preds_)
        grad, hess = self.grad_hess(y_true=targets_, y_pred=preds_)
        grad, hess = grad * weights_, hess * weights_
        # NOTE: in catboost, the definition of loss is the inverse
        grad, hess = grad * -self.grad_hess_sign, hess * -self.grad_hess_sign
        return list(zip(grad, hess))

    @final
    def is_max_optimal(self) -> bool:
        """Catboost-compatible interface"""
        return self.is_higher_better

    @final
    def evaluate(
        self,
        approxes: Sequence[float],
        target: Sequence[float],
        weight: Sequence[float] | None = None,
    ) -> tuple[float, float]:
        """Catboost-compatible interface"""
        approxes_ = np.array(approxes[0])
        targets_ = np.array(target)
        weights_ = np.array(weight) if weight is not None else np.ones_like(approxes_)
        loss = self.loss(y_true=targets_, y_pred=approxes_)
        if isinstance(loss, float) and not np.allclose(weights_, 1.0):
            warnings.warn("loss() should return ndarray when weight is not all 1.0")
            return loss, np.nan
        return float(np.sum(loss * weights_)), float(np.sum(weights_))

    @final
    def get_final_error(self, error: float, weight: float | None = None) -> float:
        """Catboost-compatible interface"""
        return error / (weight + 1e-38) if weight is not None else error

    @final
    def eval_metric_lgb(
        self,
        y_true: NDArray | lgb.Dataset | xgb.DMatrix,
        y_pred: NDArray | lgb.Dataset | xgb.DMatrix,
    ) -> tuple[str, float, bool]:
        """Sklearn-compatible interface (LightGBM)"""
        if isinstance(y_pred, lgb.Dataset) or isinstance(y_pred, xgb.DMatrix):
            # NOTE: swap (it is so fucking that the order is inconsistent)
            y_true, y_pred = y_pred, y_true
        y_true, weight = _dataset_to_ndarray(y=y_true)
        y_pred, _ = _dataset_to_ndarray(y=y_pred)
        loss = self.loss(y_true=y_true, y_pred=y_pred)
        if isinstance(loss, float) and not np.allclose(weight, 1.0):
            warnings.warn("loss() should return ndarray when weight is not all 1.0")
            return self.name, loss, self.is_higher_better
        return (
            self.name,
            float(np.sum(loss * weight) / (np.sum(weight) + 1e-38)),
            self.is_higher_better,
        )

    @final
    def eval_metric_xgb_native(
        self,
        y_true: NDArray | lgb.Dataset | xgb.DMatrix,
        y_pred: NDArray | lgb.Dataset | xgb.DMatrix,
    ) -> tuple[str, float]:
        result = self.eval_metric_lgb(y_true=y_true, y_pred=y_pred)
        return result[0], result[1]

    @final
    def eval_metric_xgb_sklearn(
        self,
        y_true: NDArray | lgb.Dataset | xgb.DMatrix,
        y_pred: NDArray | lgb.Dataset | xgb.DMatrix,
    ) -> float:
        result = self.eval_metric_lgb(y_true=y_true, y_pred=y_pred)
        return result[1]

    def __add__(self, other: LossBase) -> LossBase:
        if not isinstance(other, LossBase):
            return NotImplemented  # type: ignore
        return _LossSum(self, other)

    def __sub__(self, other: LossBase) -> LossBase:
        return self.__add__(-other)

    def __mul__(self, other: float | int | Real) -> LossBase:
        if not isinstance(other, Real):
            return NotImplemented
        return _LossMul(self, other)

    def __div__(self, other: float | int | Real) -> LossBase:
        return self.__mul__(1.0 / other)

    def __neg__(self) -> LossBase:
        return self.__mul__(-1.0)

    def __pos__(self) -> Self:
        return self


@attrs.frozen()
class _LossSum(LossBase):
    loss1: LossBase
    loss2: LossBase

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray | float:
        return self.loss1.loss(y_true=y_true, y_pred=y_pred) + self.loss2.loss(
            y_true=y_true, y_pred=y_pred
        )

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return self.loss1.grad(y_true=y_true, y_pred=y_pred) + self.loss2.grad(
            y_true=y_true, y_pred=y_pred
        )

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return self.loss1.hess(y_true=y_true, y_pred=y_pred) + self.loss2.hess(
            y_true=y_true, y_pred=y_pred
        )

    def grad_hess(self, y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        grad1, hess1 = self.loss1.grad_hess(y_true=y_true, y_pred=y_pred)
        grad2, hess2 = self.loss2.grad_hess(y_true=y_true, y_pred=y_pred)
        return grad1 + grad2, hess1 + hess2


@attrs.frozen()
class _LossMul(LossBase):
    loss_: LossBase
    factor: float | int | Real

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray | float:
        return self.factor * self.loss_.loss(y_true=y_true, y_pred=y_pred)

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return self.factor * self.loss_.grad(y_true=y_true, y_pred=y_pred)

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return self.factor * self.loss_.hess(y_true=y_true, y_pred=y_pred)

    def grad_hess(self, y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        grad, hess = self.loss_.grad_hess(y_true=y_true, y_pred=y_pred)
        return self.factor * grad, self.factor * hess
