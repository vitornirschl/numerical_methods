# TODO: Implementar o método de eliminação de Gauss

import numpy as np
from typing import Optional
import numpy.typing as npt


class Jacobi:
    def __init__(self, tolerance: float = 1e-5, max_iterations: int = 1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self.x_ = None
        self.n_iter_ = 0
        self.final_residual_ = None
        self.convergence_history_ = []

    def solve(
        self, A: npt.NDArray, b: npt.NDArray, x0: Optional[npt.NDArray] = None
    ) -> npt.NDArray:
        n_dims = len(b)
        if A.shape != (n_dims, n_dims):
            raise ValueError(
                "A must be a square matrix and match vector b's dimensions"
            )
        if np.any(np.diag(A) == 0):
            raise np.linalg.LinalgError(f"Zero detected on diagonal of matrix A.")

        if x0 is None:
            x = np.zeros(n_dims)
        else:
            x = np.array(x0, dtype=float)

        self.convergence_history_ = []

        for k in range(self.max_iterations):
            x_old = x.copy()
            for i in range(n_dims):
                sum1 = np.dot(A[i, :i], x_old[:i])
                sum2 = np.dot(A[i, i + 1 :], x_old[i + 1 :])
                x[i] = (b[i] - sum1 - sum2) / A[i, i]

            difference_norm = np.linalg.norm(x - x_old, ord=np.inf)
            self.convergence_history_.append(difference_norm)

            if difference_norm < self.tolerance:
                self.x_ = x
                self.n_iter_ = k + 1
                self.final_residual_ = np.linalg.norm(A @ x - b)
                return self.x_

        raise np.linalg.LinAlgError(
            f"Jacobi failed to converge in {self.max_iterations} iterations."
        )


class GaussSeidel:
    def __init__(self, tolerance: float = 1e-5, max_iterations: int = 1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self.x_ = None
        self.n_iter_ = 0
        self.final_residual_ = None
        self.convergence_history_ = []

    def solve(
        self, A: npt.NDArray, b: npt.NDArray, x0: Optional[npt.NDArray] = None
    ) -> npt.NDArray:
        n_dims = len(b)
        if A.shape != (n_dims, n_dims):
            raise ValueError(
                "A must be a square matrix and match vector b's dimensions"
            )
        if np.any(np.diag(A) == 0):
            raise np.linalg.LinAlgError(f"Zero detected on diagonal of matrix A.")

        if x0 is None:
            x = np.zeros(n_dims)
        else:
            x = np.array(x0, dtype=float)

        self.convergence_history_ = []

        for k in range(self.max_iterations):
            x_old = x.copy()
            for i in range(n_dims):
                sum1 = np.dot(A[i, :i], x[:i])
                sum2 = np.dot(A[i, i + 1 :], x_old[i + 1 :])
                x[i] = (b[i] - sum1 - sum2) / A[i, i]

            difference_norm = np.linalg.norm(x - x_old, ord=np.inf)
            self.convergence_history_.append(difference_norm)

            if difference_norm < self.tolerance:
                self.x_ = x
                self.n_iter_ = k + 1
                self.final_residual_ = np.linalg.norm(A @ x - b)
                return self.x_

        raise np.linalg.LinAlgError(
            f"Gauss-Seidel failed to converge in {self.max_iterations} iterations."
        )
