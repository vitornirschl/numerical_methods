import numpy as np
import numpy.typing as npt
from typing import Callable, Optional, Any
from numerical_methods.methods import OneStepMethod
from numerical_methods.protocols import RootSolver, ExplicitStepper
from numerical_methods.fixed_point import NewtonRaphson
from numerical_methods.explicit_one_step import EulerMethod


class Trapezoid(OneStepMethod):
    def __init__(
        self,
        function: Callable[..., npt.NDArray | float],
        step_size: float,
        root_solver: Optional[RootSolver] = None,
        predictor: Optional[ExplicitStepper] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(function, step_size, *args, **kwargs)

        if root_solver is None:
            self.root_solver = NewtonRaphson()
        else:
            if not isinstance(root_solver, RootSolver):
                raise TypeError("root_solver must implement RootSolver protocol")
            self.root_solver = root_solver

        if predictor is None:
            self.predictor = EulerMethod(function, step_size, *args, **kwargs)
        else:
            if not isinstance(predictor, ExplicitStepper):
                raise TypeError("predictor must implement ExplicitStepper protocol")
            self.predictor = predictor

    def _integrate(
        self, x_start: float, x_end: float, y_start: npt.NDArray, h: float
    ) -> tuple[npt.NDArray, npt.NDArray]:
        n_steps = int(np.round((x_end - x_start) / h)) + 1
        x = np.linspace(x_start, x_end, n_steps)

        n_dims = y_start.shape[0]
        y = np.zeros(shape=(n_steps, n_dims))
        y[0] = y_start

        for i in range(n_steps - 1):
            f_curr = self.function(x[i], y[i], *self.args, **self.kwargs)
            explicit_term = y[i] + (h / 2) * f_curr

            def residual_function(y_next):
                f_next = self.function(x[i + 1], y_next, *self.args, **self.kwargs)
                return y_next - explicit_term - (h / 2) * f_next

            guess = self.predictor.step(
                self.function, x[i], y[i], h, *self.args, **self.kwargs
            )

            try:
                y[i + 1] = self.root_solver.solve(residual_function, x0=guess)
            except Exception as e:
                raise RuntimeError(
                    f"Implicit solver failed at step {i} (t={x[i]:.4f})"
                ) from e

        return x, y
