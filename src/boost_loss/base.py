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


def _get_name_from_callable(obj: Callable[..., Any]) -> str:
    if hasattr(obj, "__name__"):
        return getattr(obj, "__name__")
    if hasattr(obj, "__class__") and hasattr(getattr(obj, "__class__"), "__name__"):
        return getattr(getattr(obj, "__class__"), "__name__")
    raise ValueError(f"Could not get name from callable {obj}")


class LossBase(metaclass=ABCMeta):
    """Base class for loss functions.
    Inherit this class to implement custom loss function.

    See Also
    --------
    Catboost:
    https://catboost.ai/en/docs/concepts/python-usages-examples#user-defined-loss-function

    LightGBM:
    https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#custom-objective-function

    XGBoost:
    https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html

    Example
    -------
    >>> from boost_loss.base import LossBase
    >>> import numpy as np
    >>> from numpy.typing import NDArray
    >>>
    >>> class L2Loss(LossBase):
    >>>     def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
    >>>         return (y_true - y_pred) ** 2
    >>>     def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray: # dL/dy_pred
    >>>         return -2 * (y_true - y_pred) # or (y_pred - y_true)
    >>>     def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray: # d2L/dy_pred2
    >>>         return 2 * np.ones_like(y_true) # or np.ones_like(y_true)
    >>>
    >>> from boost.sklearn import apply_custom_loss
    >>> import lightgbm as lgb
    >>> apply_custom_loss(lgb.LGBMRegressor(), L2Loss()).fit(X, y)
    """

    is_higher_better: bool = False
    """Whether the result of loss function is better when it is higher."""

    @classmethod
    @final
    def from_callable(
        cls,
        loss: Callable[[NDArray, NDArray], NDArray | float],
        grad: Callable[[NDArray, NDArray], NDArray],
        hess: Callable[[NDArray, NDArray], NDArray],
        name: str | None = None,
        is_higher_better: bool = False,
    ) -> type[Self]:
        """Create this class from loss, grad, and hess callables.

        Parameters
        ----------
        loss : Callable[[NDArray, NDArray], NDArray | float]
            The loss function. If 1-D array is returned,
            the mean of array is calculated.
            Return 1-D array if possible in order to utilize weights in the dataset
            if available.
            (y_true, y_pred) -> loss
        grad : Callable[[NDArray, NDArray], NDArray]
            The 1st order derivative (gradient) of loss w.r.t. y_pred.
            (y_true, y_pred) -> grad
        hess : Callable[[NDArray, NDArray], NDArray]
            The 2nd order derivative (Hessian) of loss w.r.t. y_pred.
            (y_true, y_pred) -> hess
        name : str | None, optional
            The name of loss function.
            If None, it tries to infer from loss function, by default None
        is_higher_better : bool, optional
            Whether the result of loss function is better when it is higher,
            by default False

        Returns
        -------
        type[Self]
            The subclass of this class.

        Raises
        ------
        ValueError
            If name is None and it can't infer from loss function.
        """
        if name is None:
            try:
                name = _get_name_from_callable(loss)
            except ValueError as e:
                raise ValueError(
                    "Could not infer name from loss function. Please specify name."
                ) from e
        return type(
            name,
            (cls,),
            dict(
                loss=staticmethod(loss),
                grad=staticmethod(grad),
                hess=staticmethod(hess),
                is_higher_better=is_higher_better,
            ),
        )

    @property
    def _grad_hess_sign(self) -> int:
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
        """Name of loss function.

        Returns
        -------
        str
            Snake case of class name. e.g. `LogCoshLoss` -> `log_cosh_loss`.
        """
        return humps.decamelize(self.__class__.__name__.replace("Loss", ""))

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        """The 1st order derivative (gradient) of loss w.r.t. y_pred.

        Parameters
        ----------
        y_true : NDArray
            The true target values.
        y_pred : NDArray
            The predicted target values.

        Returns
        -------
        NDArray
            The gradient of loss function. 1-D array with shape (n_samples,).

        Raises
        ------
        NotImplementedError
            If not implemented.
        """
        raise NotImplementedError()

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        """The 2nd order derivative (hessian) of loss w.r.t. y_pred.

        Parameters
        ----------
        y_true : NDArray
            The true target values.
        y_pred : NDArray
            The predicted target values.

        Returns
        -------
        NDArray
            The hessian of loss function. 1-D array with shape (n_samples,).

        Raises
        ------
        NotImplementedError
            If not implemented.
        """
        raise NotImplementedError()

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray | float:
        """Loss function. If 1-D array is returned, the mean of array is calculated.
        Return 1-D array if possible in order to utilize weights in the dataset
        if available.

        Parameters
        ----------
        y_true : NDArray
            The true target values.
        y_pred : NDArray
            The predicted target values.

        Returns
        -------
        NDArray | float
            The loss function. 1-D array with shape (n_samples,) or float.

        Raises
        ------
        NotImplementedError
            If not implemented.
        """
        raise NotImplementedError()

    def grad_hess(self, y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        """Gradient and hessian of loss function. Override this method if you want to
        calculate both gradient and hessian at the same time.

        Parameters
        ----------
        y_true : NDArray
            The true target values.
        y_pred : NDArray
            The predicted target values.

        Returns
        -------
        tuple[NDArray, NDArray]
            The gradient and hessian of loss function.
            1-D array with shape (n_samples,).
        """
        return self.grad(y_true=y_true, y_pred=y_pred), self.hess(
            y_true=y_true, y_pred=y_pred
        )

    def _grad_hess_weighted(
        self, y_true: NDArray, y_pred: NDArray, weight: NDArray
    ) -> tuple[NDArray, NDArray]:
        grad, hess = self.grad_hess(y_true=y_true, y_pred=y_pred)
        if np.any(hess < 0):
            negative_rate = np.mean(hess < 0)
            warnings.warn(
                f"Found negative hessian in {negative_rate:.2%} samples."
                "This may cause convergence issue and cause CatBoostError in CatBoost."
                "If LightGBM or XGBoost is used, the estimator will return "
                "nonsense values (like all 0s if 100%).",
                RuntimeWarning,
            )
        grad, hess = grad * weight, hess * weight
        grad, hess = grad * self._grad_hess_sign, hess * self._grad_hess_sign
        return grad, hess

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
        return self._grad_hess_weighted(y_true=y_true, y_pred=y_pred, weight=weight)

    @final
    def calc_ders_range(
        self,
        preds: Sequence[float],
        targets: Sequence[float],
        weights: Sequence[float] | None = None,
    ) -> list[tuple[float, float]]:
        """Catboost-compatible interface"""
        y_pred = np.array(preds)
        y_true = np.array(targets)
        weight = np.array(weights) if weights is not None else np.ones_like(y_pred)
        grad, hess = self._grad_hess_weighted(
            y_true=y_true, y_pred=y_pred, weight=weight
        )
        # NOTE: in catboost, the definition of loss is the inverse
        grad, hess = -grad, -hess
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
        """LightGBM-compatible interface"""
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
        """XGBoost-native-api-compatible interface"""
        result = self.eval_metric_lgb(y_true=y_true, y_pred=y_pred)
        return result[0], result[1]

    @final
    def eval_metric_xgb_sklearn(
        self,
        y_true: NDArray | lgb.Dataset | xgb.DMatrix,
        y_pred: NDArray | lgb.Dataset | xgb.DMatrix,
    ) -> float:
        """XGBoost-sklearn-api-compatible interface"""
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

    def __radd__(self, other: LossBase) -> LossBase:
        return self.__add__(other)

    def __rsub__(self, other: LossBase) -> LossBase:
        return self.__sub__(other)

    def __rmul__(self, other: float | int | Real) -> LossBase:
        return self.__mul__(other)

    def __rdiv__(self, other: float | int | Real) -> LossBase:
        return self.__div__(other)

    def __neg__(self) -> LossBase:
        return self.__mul__(-1.0)

    def __pos__(self) -> Self:
        return self


@attrs.define()
class _LossSum(LossBase):
    loss1: LossBase
    loss2: LossBase

    @property
    def name(self) -> str:
        return self.loss1.name + "+" + self.loss2.name

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


@attrs.define()
class _LossMul(LossBase):
    loss_: LossBase
    factor: float | int | Real

    @property
    def name(self) -> str:
        return f"{self.factor}*{self.loss_.name}"

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray | float:
        return self.factor * self.loss_.loss(y_true=y_true, y_pred=y_pred)

    def grad(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return self.factor * self.loss_.grad(y_true=y_true, y_pred=y_pred)

    def hess(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        return self.factor * self.loss_.hess(y_true=y_true, y_pred=y_pred)

    def grad_hess(self, y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        grad, hess = self.loss_.grad_hess(y_true=y_true, y_pred=y_pred)
        return self.factor * grad, self.factor * hess
