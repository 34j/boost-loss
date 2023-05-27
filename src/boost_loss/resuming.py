from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .base import LossBase


class ResumingLoss(LossBase):
    def __init__(
        self,
        losses: Sequence[LossBase],
        *,
        weights: Sequence[float] | None = None,
        interval: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.losses = losses
        if weights is None:
            self.weights = np.ones_like(losses)
        else:
            self.weights = np.array(weights)
        self.interval = interval
        self.random_state = random_state
        if self.random_state is None:
            if weights is not None:
                raise ValueError("weights must be None when random_state is None")
        else:
            self.random = np.random.RandomState(self.random_state)
        self._count = 0
        self._idx = 0

    def grad_hess(self, y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        if self._count % self.interval == 0:
            if self.random_state is None:
                self._idx = self.random.choice(len(self.losses), p=self.weights)
            else:
                self._idx = (self._count // self.interval) % len(self.losses)
        self._count += 1
        return self.losses[self._idx].grad_hess(y_true=y_true, y_pred=y_pred)

    def loss(self, y_true: NDArray, y_pred: NDArray) -> NDArray | float:
        return self.losses[self._idx].loss(y_true=y_true, y_pred=y_pred)
