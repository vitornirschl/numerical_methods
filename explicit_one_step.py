import numpy as np
import numpy.typing as npt
from typing import Callable


class Euler:
    def __init__(self):
        self.function = None
        self.x = None
        self.y = None

    def fit(
        self,
        function: Callable[[float, npt.NDArray], npt.NDArray],
        y_start: npt.NDArray,
        x_start: float,
        x_end: float,
        n_steps: int,
        *args,
        **kwargs
    ) -> None:

        self.function = function
        self.x = np.linspace(x_start, x_end, n_steps + 1)

        n_dims = len(y_start)
        self.y = np.zeros((n_steps + 1, n_dims))
        self.y[0] = y_start

        h = (x_end - x_start) / n_steps

        for i in range(1, n_steps + 1):
            self.y[i] = self.y[i - 1] + h * self.function(
                self.x[i - 1], self.y[i - 1], *args, **kwargs
            )


# TODO: Quero que seja um método multivariável, recebendo um numpy array igual o interpolation
# TODO: Fazer as funções de teste e verificar se a divisão dos intervalos tá bem feita. Preciso garantir que x[-1] = x_end.
