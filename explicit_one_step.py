import numpy as np
import numpy.typing as npt
from typing import Callable, Optional, Any
from numerical_methods.methods import OneStepMethod
from numerical_methods.protocols import ExplicitStepper
from numerical_methods.utils import TableResult


class EulerMethod(OneStepMethod):
    def _integrate(
        self, x_start: float, x_end: float, y_start: npt.NDArray | float, h: float
    ):
        n_steps = int(np.round((x_end - x_start) / h)) + 1

        x = np.linspace(x_start, x_end, n_steps)
        n_dims = y_start.shape[0]
        y = np.zeros(shape=(n_steps, n_dims))
        y[0] = y_start

        for i in range(n_steps - 1):
            y[i + 1] = self.step(
                self.function, x[i], y[i], h, *self.args, **self.kwargs
            )

        return x, y

    def step(
        self,
        function: Callable[..., npt.NDArray],
        x: float,
        y: npt.NDArray,
        step_size: float,
        *args: Any,
        **kwargs: Any
    ) -> npt.NDArray:
        return y + step_size * function(x, y, *args, **kwargs)
