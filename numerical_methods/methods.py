from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from typing import Callable, Optional, Any

from .utils import TableResult


class OneStepMethod(ABC):
    def __init__(
        self,
        function: Callable[..., npt.NDArray | float],
        step_size: float,
        *args,
        **kwargs,
    ) -> None:
        if not isinstance(function, Callable):
            raise TypeError("function must be a callable object")
        if not isinstance(step_size, (int, float)):
            raise TypeError("step_size must be a number")
        if step_size <= 0:
            raise ValueError("step_size must be positive")

        self.function = function
        self.step_size = step_size
        self.args = args
        self.kwargs = kwargs

        self.x_: Optional[npt.NDArray] = None
        self.y_: Optional[npt.NDArray] = None

    @abstractmethod
    def _integrate(
        self, x_start: float, x_end: float, y_start: npt.NDArray, h: float
    ) -> tuple[npt.NDArray, npt.NDArray]:
        pass

    def solve(
        self, x_start: float, x_end: float, y_start: npt.NDArray | float
    ) -> tuple[npt.NDArray, npt.NDArray]:
        if not all(isinstance(x, (int, float)) for x in [x_start, x_end]):
            raise TypeError("x_start and x_end must be numbers (int or float)")
        if not isinstance(y_start, (int, float, np.ndarray)):
            raise TypeError("y_start must be a number or a numpy array")
        if x_end < x_start:
            raise ValueError("x_end must be greater than x_start")

        y_start_arr = np.atleast_1d(y_start).astype(float)

        x, y = self._integrate(x_start, x_end, y_start_arr, self.step_size)

        self.x_ = x
        self.y_ = y.squeeze()

        return self.x_, self.y_

    def convergence_table(
        self,
        x_start: float,
        x_end: float,
        y_start: npt.NDArray | float,
        y_exact: Optional[Callable[..., npt.NDArray | float]] = None,
        first_n: int = 2,
        n_rows: int = 8,
    ) -> TableResult:
        if not isinstance(y_start, (int, float, np.ndarray)):
            raise TypeError("y_start must be a number or a numpy array")
        if y_exact is not None:
            if not isinstance(y_exact, Callable):
                raise TypeError("y_exact must be a callable object")
        if not all(isinstance(x, (int, float)) for x in [x_start, x_end]):
            raise TypeError("x_start and x_end must be numbers (int or float)")
        if not all(isinstance(n, int) for n in [first_n, n_rows]):
            raise TypeError("first_n and n_rows must be integers")
        if not all(n > 0 for n in [first_n, n_rows]):
            raise ValueError("first_n and n_rows must be positive integers")

        y_start = np.atleast_1d(y_start).astype(float)

        if y_exact is not None:
            y_true_end = np.atleast_1d(y_exact(x_end))

        results = []
        last_metric = 0.0

        y_end_prev = None

        for i in range(n_rows):
            n_steps = first_n * (2**i)
            h = (x_end - x_start) / n_steps

            _, y_arr = self._integrate(x_start, x_end, y_start, h)
            y_end_curr = y_arr[-1]

            if y_exact is not None:
                metric = np.linalg.norm(y_end_curr - y_true_end)
                col_name = "Error (Exact)"
            else:
                if y_end_prev is not None:
                    metric = np.linalg.norm(y_end - y_end_prev)
                else:
                    metric = 0.0
                col_name = "Diff (y_h - y_2h)"

            q = last_metric / metric if i > 0 and metric != 0 else 0.0
            log2_q = np.log2(q) if q > 0 else 0.0

            results.append([n_steps, h, metric, q, log2_q])

            last_metric = metric
            y_end_prev = y_end_curr

        return TableResult(
            data=results,
            headers=["n", "h", col_name, "q", "log2(q)"],
            float_formats=(".0f", ".3e", ".3e", ".3e", ".3f"),
        )


class IterativeLinearSolver(ABC):
    def __init__(self, tolerance: float = 1e-5, max_iterations: int = 1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self.x_ = None
        self.n_iter_ = 0
        self.final_residual_ = None
        self.convergence_history_ = []

    @abstractmethod
    def solve(
        self, A: npt.NDArray, b: npt.NDArray, x0: Optional[npt.NDArray] = None
    ) -> npt.NDArray:
        pass

    def convergence_table(self) -> TableResult:
        if not self.convergence_history_:
            raise ValueError("The solver has not been run yet. Call solve() first.")

        data = []
        headers = ["Iteration", "||x_new - x_old||", "Ratio"]

        for k, error in enumerate(self.convergence_history_):
            ratio = 0.0
            if k > 0 and self.convergence_history_[k - 1] > 0:
                ratio = error / self.convergence_history_[k - 1]
            data.append([k + 1, error, ratio])

        return TableResult(
            data=data,
            headers=headers,
            float_formats=(".0f", ".4e", ".4f"),
        )
