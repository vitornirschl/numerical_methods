from typing import Callable
import numpy as np
import numpy.typing as npt


class NotFittedError(ValueError):
    """
    Exception for methods called before fit().
    """

    def __init__(
        self,
        msg: str = "This instance was not fitted yet. Call the method fit() first.",
    ) -> None:
        super().__init__(msg)


class Splines:
    """
    Natural cubic splines interpolation class.

    Attributes
    ----------
    x_ : npt.ArrayLike
        Sorted independent variable data from `fit`.
    y_ : npt.ArrayLike
        Sorted dependent variable data from `fit`.
    coeffs_ : npt.ArrayLike
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
        x = np.asarray(x)
        y = np.asarray(y)

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
        b[-1] = 0
        c[-1] = 0
        d[-1] = 0

        h = np.diff(x)
        alpha = np.zeros(n)

        el = np.zeros(n + 1)
        el[0] = 1
        el[-1] = 1

        z = np.zeros(n + 1)
        z[0] = 0
        z[-1] = 0

        mu = np.zeros(n)
        mu[0] = 0

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

    def predict(self, x: npt.ArrayLike) -> npt.NDArray:
        """
        Predicts the dependent variable at the given independent variable.

        Args
        ----
        x : npt.ArrayLike
            Independent variable(s). Can be a single float or an array of floats.

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

if __name__ == "__main__":
    import matplotlib

    try:
        matplotlib.use("TkAgg")  # Tenta forçar o backend Tk
        print("Backend do Matplotlib forçado para 'TkAgg'")
    except ImportError:
        print("Backend 'TkAgg' não encontrado, usando o padrão.")

    import matplotlib.pyplot as plt

    def testar_e_plotar_funcao(
        ax: plt.Axes,
        func: Callable,
        x_min: float,
        x_max: float,
        titulo: str,
        num_nos: int = 10,
        num_plot: int = 200,
    ):
        """
        Função auxiliar para treinar, prever e plotar uma função
        e sua interpolação por spline.
        """
        print(f"Testando: {titulo} com {num_nos} nós...")

        # 1. Gerar dados de treino (nós)
        x_treino = np.linspace(x_min, x_max, num_nos)
        y_treino = func(x_treino)

        # 2. Criar e treinar o modelo Spline
        modelo_spline = Splines()
        modelo_spline.fit(x_treino, y_treino)

        # 3. Gerar dados de plotagem (bem densos para uma curva suave)
        x_plot = np.linspace(x_min, x_max, num_plot)
        y_real = func(x_plot)

        # 4. Obter predições do spline
        y_pred = modelo_spline.predict(x_plot)

        # 5. Calcular o erro
        # Usamos o erro dos pontos densos para uma boa estimativa
        mse = np.mean((y_real - y_pred) ** 2)

        # 6. Plotar tudo no eixo (ax) fornecido
        ax.plot(x_plot, y_real, "b--", label="Função Real", alpha=0.7)
        ax.plot(x_plot, y_pred, "r-", label="Interpolação Spline", linewidth=2)
        ax.scatter(
            x_treino, y_treino, color="black", zorder=5, label=f"Nós (N={num_nos})"
        )

        ax.set_title(f"{titulo}\nErro Quadrático Médio (MSE): {mse:.2e}")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

    # --- Configuração dos Testes ---

    # Define as funções que queremos testar
    funcoes_para_testar = [
        {
            "func": np.sin,
            "x_min": -np.pi,
            "x_max": np.pi,
            "titulo": "f(x) = sin(x)",
            "num_nos": 8,
        },
        {
            "func": lambda x: 1 / (1 + 25 * x**2),  # Famosa Função de Runge
            "x_min": -1.0,
            "x_max": 1.0,
            "titulo": "Função de Runge",
            "num_nos": 15,
        },
        {
            "func": lambda x: np.cos(x) + np.sin(2 * x) / 2,
            "x_min": 0,
            "x_max": 2 * np.pi,
            "titulo": "f(x) = cos(x) + sin(2x)/2",
            "num_nos": 12,
        },
        {
            # Um polinômio cúbico. A interpolação deve ser *exata*.
            # Ótimo "teste de sanidade" para o seu código!
            "func": lambda x: 0.1 * x**3 - 0.5 * x**2 + x - 2,
            "x_min": -5.0,
            "x_max": 5.0,
            "titulo": "Polinômio Cúbico (Teste de Sanidade)",
            "num_nos": 5,  # Um spline cúbico deve ser exato com >= 4 pontos
        },
    ]

    # Cria a figura e os subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()  # Facilita o loop

    for i, config in enumerate(funcoes_para_testar):
        testar_e_plotar_funcao(
            ax=axes[i],
            func=config["func"],
            x_min=config["x_min"],
            x_max=config["x_max"],
            titulo=config["titulo"],
            num_nos=config["num_nos"],
        )

    fig.suptitle("Comparação da Interpolação com Splines Cúbicos Naturais", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajusta para o super-título
    plt.show()
