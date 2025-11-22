import numpy as np
import numpy.typing as npt
import numdifftools as nd
from numerical_methods.linalg import GaussSeidel
from numerical_methods.protocols import LinearSolver
from typing import Callable, Optional


class NewtonRaphson:
    def __init__(
        self,
        tolerance: float = 1e-5,
        max_iterations: int = 1000,
        linear_solver: Optional[LinearSolver] = None,
    ):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        if linear_solver is None:
            internal_tolerance = tolerance * 0.1
            self.linear_solver = GaussSeidel(
                tolerance=internal_tolerance, max_iterations=100
            )
        else:
            if not isinstance(linear_solver, LinearSolver):
                raise TypeError(
                    "The given 'linear_solver' object does not implement the LinearSolver protocol"
                )
            self.linear_solver = linear_solver

        self.x_ = None
        self.n_iter_ = 0
        self.final_residual_ = None
        self.convergence_history_ = []

    def solve(
        self,
        function: Callable[..., npt.NDArray],
        x0: npt.NDArray,
        jacobian: Optional[Callable[..., npt.NDArray]] = None,
        *args,
        **kwargs,
    ) -> npt.NDArray:

        n_dims = len(x0)
        x = np.array(x0, dtype=float)

        calc_F = lambda y: function(y, *args, **kwargs)

        if jacobian is None:
            calc_J = nd.Jacobian(calc_F)
        else:
            calc_J = lambda y: jacobian(y, *args, **kwargs)

        self.convergence_history_ = []

        for k in range(self.max_iterations):
            F = calc_F(x)
            J = calc_J(x)

            dx_guess = np.zeros(n_dims)
            dx = self.linear_solver.solve(J, -F, x0=dx_guess)
            x += dx

            difference_norm = np.linalg.norm(dx, ord=np.inf)
            self.convergence_history_.append(difference_norm)

            if difference_norm < self.tolerance:
                self.x_ = x
                self.n_iter_ = k + 1

                F_final = calc_F(x)
                self.final_residual_ = np.linalg.norm(F_final)

                return self.x_

        raise np.linalg.LinAlgError(
            f"Newton-Raphson failed to converge in {self.max_iterations} iterations."
        )
