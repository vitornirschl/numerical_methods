from typing import Callable, Optional
import numpy as np
import numpy.typing as npt
from numerical_methods.utils import NotFittedError


class NaturalCubicSplines:
    """
    Natural cubic splines interpolation class.

    Attributes
    ----------
    x_ : npt.ArrayLike
        Sorted independent variable data from `fit`.
    y_ : npt.ArrayLike
        Sorted dependent variable data from `fit`.
    coeffs_ : npt.NDArray
        Computed spline coefficients (shape 4 x n_splines).

    Methods
    -------
    fit(x, y)
        Fit the splines to the data.
    predict(x)
        Predict the dependent variable at the given independent variable.
    """

    def __init__(self):
        self.x_ = None
        self.y_ = None
        self.coeffs_ = None
        self._n_splines = 0

    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> None:
        """
        Fit the splines to the data.

        Parameters
        ----------
        x : npt.ArrayLike
            Independent  variable data.
        y : npt.ArrayLike
            Dependent variable data.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If x and y have different dimensions.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.shape != y.shape:
            msg = "x and y must have the same dimensions"
            raise ValueError(msg)

        sorted_indices = x.argsort()
        x = x[sorted_indices]
        y = y[sorted_indices]

        n = len(x) - 1

        a = y
        b = np.zeros(n + 1)
        c = np.zeros(n + 1)
        d = np.zeros(n + 1)

        h = np.diff(x)
        alpha = np.zeros(n)

        el = np.zeros(n + 1)
        el[0] = 1
        el[-1] = 1

        z = np.zeros(n + 1)

        mu = np.zeros(n)

        for i in range(1, n):
            alpha[i] = 3 / h[i] * (a[i + 1] - a[i]) - 3 / h[i - 1] * (a[i] - a[i - 1])

            el[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]

            mu[i] = h[i] / el[i]

            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / el[i]

        for i in range(n - 1, -1, -1):
            c[i] = z[i] - mu[i] * c[i + 1]

            b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3

            d[i] = (c[i + 1] - c[i]) / (3 * h[i])

        self.x_ = x
        self.y_ = y
        self.coeffs_ = np.asarray([a[:n], b[:n], c[:n], d[:n]])
        self._n_splines = n

    def predict(
        self, x: npt.ArrayLike, allow_extrapolation: Optional[bool] = False
    ) -> npt.NDArray:
        """
        Predicts the dependent variable at the given independent variable.

        Parameters
        ----------
        x : npt.ArrayLike
            Independent variable(s). Can be a single float or an array of floats.
        allow_extrapolation : bool
            If True, allows x to be outside the fitted range, using the first or
            the last spline polynomial for extrapolation.
            If False (default), raises a ValueError for out-of-range values.

        Returns
        -------
        npt.NDArray
            Predicted dependent variable(s).

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        ValueError
            If x is not within the fitted range.
        """
        if self.coeffs_ is None:
            raise NotFittedError()

        x = np.asarray(x)
        is_scalar = x.ndim == 0
        x = np.atleast_1d(x)

        x0 = self.x_[0]
        xn = self.x_[-1]

        if not allow_extrapolation:
            if np.any(x < x0) or np.any(x > xn):
                raise ValueError(
                    f"x values must be within the fitted range [{x0:.1f}, {xn:.1f}]."
                )

        i = np.searchsorted(self.x_, x, side="right") - 1
        i = np.clip(i, 0, self._n_splines - 1)

        a = self.coeffs_[0, i]
        b = self.coeffs_[1, i]
        c = self.coeffs_[2, i]
        d = self.coeffs_[3, i]

        dx = x - self.x_[i]
        y_pred = a + b * dx + c * dx**2 + d * dx**3

        if is_scalar:
            return y_pred.item()
        return y_pred


# -----------------------------------------------------------------
