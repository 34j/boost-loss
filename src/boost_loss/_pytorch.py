from abc import ABCMeta
from typing import Any, Callable, final

import attrs
import numpy as np
import torch
from numpy.typing import NDArray

from ._base import LossBase


class TorchLossBase(LossBase, metaclass=ABCMeta):
    """Calculate gradient and hessian using `torch.autograd.grad`.
    One of `loss_torch` and `grad_torch` must be implemented.

    Inspired by
    https://github.com/TomerRonen34/treeboost_autograd/blob/main/treeboost_autograd/pytorch_objective.py
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        loss_inherited = cls.loss_torch is not TorchLossBase.loss_torch
        grad_inherited = cls.grad_torch is not TorchLossBase.grad_torch
        if loss_inherited or grad_inherited:
            pass
        else:
            raise TypeError(
                f"Can't instantiate abstract class {cls.__name__} "
                "with loss_torch or grad_torch not implemented"
            )
        return super().__init_subclass__(**kwargs)

    def loss_torch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Parameters
        ----------
        y_true : torch.Tensor
            The true target values.
        y_pred : torch.Tensor
            The predicted target values.

        Returns
        -------
        torch.Tensor
            0-dim or 1-dim tensor of shape (n_samples,). Return 1-dim tensor if possible
            to utilize weights in the dataset if available.

        Raises
        ------
        NotImplementedError
            If not implemented.
        """
        raise NotImplementedError()

    def grad_torch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """The 1st order derivative of loss w.r.t. y_pred.

        Parameters
        ----------
        y_true : torch.Tensor
            The true target values.
        y_pred : torch.Tensor
            The predicted target values.

        Returns
        -------
        torch.Tensor
            The gradient of loss w.r.t. y_pred. 1-dim tensor of shape (n_samples,).

        Raises
        ------
        NotImplementedError
            If not implemented.
        """
        raise NotImplementedError()

    @final
    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray | float:
        loss = self.loss_torch(
            torch.from_numpy(y_true), torch.from_numpy(y_pred)
        ).detach()
        if loss.ndim == 0:
            return loss.item()
        else:
            return loss.numpy()

    @final
    def grad_hess(self, y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        y_true_ = torch.from_numpy(y_true)
        y_pred_ = torch.from_numpy(y_pred).requires_grad_(True)

        try:
            grad = self.grad_torch(y_true_, y_pred_)
        except NotImplementedError:
            loss = torch.mean(self.loss_torch(y_true_, y_pred_)) * y_true_.size(0)
            grad = torch.autograd.grad(loss, y_pred_, create_graph=True)[0]
            hess = np.array(
                [
                    torch.autograd.grad(grad_, y_pred_, retain_graph=True)[0][i].item()
                    for i, grad_ in enumerate(grad)
                ]
            )
        else:
            hess = np.array(
                [
                    torch.autograd.grad(grad_, y_pred_, create_graph=True)[0][i].item()
                    for i, grad_ in enumerate(grad)
                ]
            )
        return grad.detach().numpy(), hess

    @classmethod
    @final
    def from_function_torch(
        cls,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        name: str | None = None,
        is_higher_better: bool = False,
    ) -> type["TorchLossBase"]:
        if name is None and hasattr(loss, "__name__"):
            name = getattr(loss, "__name__") + f"{cls.__name__}"
        if (
            name is None
            and hasattr(loss, "__class__")
            and hasattr(getattr(loss, "__class__"), "__name__")
        ):
            name = getattr(getattr(loss, "__class__"), "__name__") + f"{cls.__name__}"
        if name is None or name == f"<lambda>{cls.__name__}":
            raise ValueError(
                "Could not infer name from loss function. Please specify name."
            )
        return type(
            name,
            (cls,),
            dict(loss_torch=staticmethod(loss), is_higher_better=is_higher_better),
        )


@attrs.define(kw_only=True)
class _LNLossTorch_(TorchLossBase):
    """L^n loss for PyTorch."""

    n: float
    divide_n_loss: bool = False
    divide_n_grad: bool = False

    def loss_torch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        if self.divide_n_grad != self.divide_n_loss:
            raise ValueError(
                "divide_n_grad and divide_n_loss must be the same, "
                f"but got {self.divide_n_grad} and {self.divide_n_loss}"
            )
        return torch.abs(y_true - y_pred) ** self.n / (
            self.n if self.divide_n_loss else 1
        )


@attrs.define(kw_only=True)
class _LNLossTorch(TorchLossBase):
    """L^n loss for PyTorch."""

    n: float
    divide_n_loss: bool = False
    divide_n_grad: bool = True

    def loss_torch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.abs(y_pred - y_true) ** self.n / (
            self.n if self.divide_n_loss else 1
        )

    def grad_torch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return (
            torch.sign(y_pred - y_true)
            * torch.abs(y_pred - y_true) ** (self.n - 1)
            * self.n
            / (self.n if self.divide_n_grad else 1)
        )
