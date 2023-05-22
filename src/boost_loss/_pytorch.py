from abc import ABCMeta
from typing import Callable
import attrs

import torch
from numpy.typing import NDArray
from typing_extensions import Self

from boost_loss._base import LossBase

from ._base import LossBase


class TorchLossBase(LossBase, metaclass=ABCMeta):
    """Calculate gradient and hessian using `torch.autograd.grad`.
    One of `loss_torch` and `grad_torch` must be implemented.

    Inspired by
    https://github.com/TomerRonen34/treeboost_autograd/blob/main/treeboost_autograd/pytorch_objective.py
    """

    def __new__(cls) -> Self:
        loss_inherited = cls.loss_torch is not TorchLossBase.loss_torch
        grad_inherited = cls.grad_torch is not TorchLossBase.grad_torch
        if loss_inherited or grad_inherited:
            pass
        else:
            raise TypeError(
                f"Can't instantiate abstract class {cls.__name__} "
                "with loss_torch or grad_torch not implemented"
            )
        return super().__new__(cls)

    def loss_torch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def grad_torch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray | float:
        return self.loss_torch(torch.from_numpy(y_true), torch.from_numpy(y_pred)).detach().numpy()

    def grad_hess(self, y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        y_true_ = torch.from_numpy(y_true).requires_grad_(True)
        y_pred_ = torch.from_numpy(y_pred)

        try:
            loss = self.loss_torch(y_true_, y_pred_)
        except NotImplementedError:
            grad = self.grad_torch(y_true_, y_pred_)
        else:
            grad = torch.autograd.grad(loss, y_true_, create_graph=True)[0]
        hess = torch.autograd.grad(grad, y_true_, create_graph=True)[0]
        return grad.detach().numpy(), hess.detach().numpy()

    @classmethod
    def from_function_torch(
        cls,
        name: str,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        is_higher_better: bool = False,
    ) -> type["TorchLossBase"]:
        return attrs.make_class(
            name,
            bases=(cls,),
            attrs=dict(
                loss_torch=loss,
                is_higher_better=is_higher_better,
            ),
        )

class _L2LossTorch(TorchLossBase):
    """L2 loss for PyTorch."""

    def loss_torch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.abs(y_true - y_pred) ** 2 / 2