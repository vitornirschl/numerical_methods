import numpy as np
import numpy.typing as npt
from typing import Callable, Literal, Optional
from tabulate import tabulate
import json


class ConvergenceTableResult:
    """
    A class for results objects that stores data from a convergence table.
    It can print beautiful tables and export the results to various formats.

    Attributes
    ----------
    data : list[list]
        Result for convergence table.
    _headings : list[str]
        Headings for the table.
    _float_formats : tuple
        Formatting for the float strings of the table.

    Methods
    -------
    export() -> str | None
        Exports the table to a file. Returns None if no filename is provided.
    __str__() -> str
        Returns a string representation of the table.
    __repr__() -> str
        Returns a string representation of the object.
    _to_typst() -> str
        Returns a Typst representation of the table.
    _to_latex() -> str
        Returns a LaTeX representation of the table.
    _to_csv() -> str
        Returns a CSV representation of the table.
    _to_json() -> str
        Returns a JSON representation of the table.
    """

    def __init__(self, data: list[list]):
        """
        Initialization method for ConvergenceTableResult class.

        Args
        ----
        data : list[list]
            Result for convergence table.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If data is not a list.
            If any row in data is not a list.
        """
        if not isinstance(data, list):
            raise TypeError("data must be a list")
        if not all(isinstance(row, list) for row in data):
            raise TypeError("data must be a list of lists")

        self.data = data
        self._headings = ["n", "h", "error", "q", "log2(q)"]
        self._float_formats = (".0f", ".3e", ".3e", ".3e", ".3f")

    def __str__(self) -> str:
        """
        Returns a string representation of the table.
        """
        return tabulate(
            self.data,
            headers=self._headings,
            floatfmt=self._float_formats,
            tablefmt="github",
            numalign="center",
            stralign="center",
        )

    def __repr__(self) -> str:
        """
        Returns a string representation of the object.
        """
        return f"<ConvergenceTableResult(n_rows={len(self.data)})>"

    def _to_typst(self) -> str:
        """
        Returns a Typst representation of the table.
        """
        # The initialization of the table
        table = "#table(\n" "   columns: 5,\n" "   align: center + horizon,\n"

        # The header of the table
        header_cells = [f"[*{h}*]" for h in self._headings]
        table += "  " + ", ".join(header_cells) + ",\n"

        # The rows of the table
        data_row_strings = []
        for row in self.data:
            cells_in_row = []
            # Pairs each value with its correct formatting
            for value, fmt in zip(row, self._float_formats):
                formatted_val = f"{value:{fmt}}"
                cells_in_row.append(f"[{formatted_val}]")
            data_row_strings.append("   " + ", ".join(cells_in_row))
        # Joins the rows to the table
        table += ",\n".join(data_row_strings)
        table += "\n)"

        return table

    def _to_latex(self) -> str:
        """
        Returns a LaTeX representation of the table.
        """
        table = tabulate(
            self.data,
            headers=self._headings,
            floatfmt=self._float_formats,
            tablefmt="latex",
            numalign="center",
            stralign="center",
        )
        return table

    def _to_csv(self) -> str:
        """
        Returns a CSV representation of the table.
        """
        # Headings of the table
        table = ",".join(self._headings) + "\n"
        data_row_strings = []
        for row in self.data:
            cells_in_row = []
            for value, fmt in zip(row, self._float_formats):
                formatted_val = f"{value:{fmt}}"
                cells_in_row.append(formatted_val)
            # Joins cells with comma and adds to the list of row strings
            data_row_strings.append(",".join(cells_in_row))
        # Joins the rows to the table
        table += "\n".join(data_row_strings)

        return table

    def _to_json(self) -> str:
        """
        Returns a JSON representation of the table.
        """
        objects = []

        for row in self.data:
            row_dict = dict(zip(self._headings, row))
            objects.append(row_dict)

        return json.dumps(objects, indent=4)

    def export(
        self, format: Literal["typst", "latex", "csv", "json"], filename: Optional[str]
    ) -> str | None:
        """
        Exports the convergence table to a string format (and optionally to a file).

        Args
        ----
        format : str
            Exporting format. Must be one of 'typst', 'latex', 'csv', or 'json'.
        filename : str, optional
            If given, saves the formatted table to a file in this path.

        Returns
        -------
        str | None
            The formatted string (if filename=None) or None (if filename is given)

        Raises
        ------
        ValueError
            If format is not one of 'typst', 'latex', 'csv', or 'json'.
        """
        msg_wrong_format = "Format must be one of 'typst', 'latex', 'csv', or 'json'."
        if format not in ["typst", "latex", "csv", "json"]:
            raise ValueError(msg_wrong_format)

        output_str = ""
        if format == "typst":
            output_str = self._to_typst()
        elif format == "latex":
            output_str = self._to_latex()
        elif format == "csv":
            output_str = self._to_csv()
        else:  # format == "json":
            output_str = self._to_json()

        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(output_str)
            print(f"Table exported to {filename}")
            return None

        return output_str


class EulerMethod:
    """
    Class implementation of the Euler method for solving ODEs.

    Attributes
    ----------
    function : Callable[..., npt.NDArray | float]
        The function that defines the system of ODEs,
        i.e. the function f(x, y) such that dy/dx = f(x, y).
    step_size : int | float
        The step size for the Euler method.
    x_ : npt.NDArray
        The array of x values.
    y_ : npt.NDArray
        The array of y values.

    Methods
    -------
    solve(x_start, x_end, y_start)
        Solve the system of ODEs using the Euler method in the interval [x_start, x_end] with
        the initial condition y_start.
    """

    def __init__(
        self,
        function: Callable[..., npt.NDArray | float],
        step_size: float,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialization method for EulerMethod class.

        Parameters
        ----------
        function : Callable[..., npt.NDArray | float]
            The function that defines the system of ODEs,
            i.e. the function f(x, y) such that dy/dx = f(x, y).
        step_size : float
            The step size for the Euler method.
        *args : tuple
            Additional positional arguments to pass to the function.
        **kwargs : dict
            Additional keyword arguments to pass to the function.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If function is not a callable object.
        TypeError
            If step_size is not a number.
        ValueError
            If step_size is not positive.
        """

        # Exception Handling
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

        self.x_ = None
        self.y_ = None

    def solve(
        self,
        x_start: float,
        x_end: float,
        y_start: npt.NDArray | float,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Solve the system of ODEs using the Euler method.

        Parameters
        ----------
        x_start : float
            The initial value of the independent variable.
        x_end : float
            The final value of the independent variable.
        y_start : npt.NDArray | float
            The initial value of the dependent variable.

        Returns
        -------
        tuple[npt.NDArray, npt.NDArray]
            A tuple containing the array of x values and the array of y values,
            where y has shape (n_steps, n_dims) or (n_steps,) for the scalar case.

        Raises
        ------
        TypeError
            If x_start or x_end is not a number.
        TypeError
            If y_start is not a number or a numpy array.
        ValueError
            If x_end is less than x_start.
        """

        # Exception Handling
        if not all(isinstance(x, (int, float)) for x in [x_start, x_end]):
            raise TypeError("x_start and x_end must be numbers (int or float)")
        if not isinstance(y_start, (int, float, np.ndarray)):
            raise TypeError("y_start must be a number or a numpy array")
        if x_end < x_start:
            raise ValueError("x_end must be greater than x_start")

        y_start = np.atleast_1d(y_start)
        n_dims = y_start.shape[0]  # The number of dimensions of the system of ODEs.
        h = self.step_size
        n_steps = int((x_end - x_start) / h) + 1

        x = np.linspace(x_start, x_end, n_steps)
        y = np.zeros(shape=(n_steps, n_dims))
        y[0] = y_start

        for i in range(n_steps - 1):
            y[i + 1] = y[i] + h * self.function(x[i], y[i], *self.args, **self.kwargs)

        self.x_ = x
        self.y_ = y.squeeze()

        return self.x_, self.y_

    def convergence_table(
        self,
        x_start: float,
        x_end: float,
        y_start: npt.NDArray | float,
        y_exact: Callable[..., npt.NDArray | float],
        first_n: int = 2,
        n_rows: int = 8,
    ) -> ConvergenceTableResult:
        """
        Generates a convergence table for the Euler method.

        Parameters
        ----------
        x_start : float
            The first value of the independent variable.
        x_end : float
            The last value of the independent variable.
        y_start : npt.NDArray | float
            The initial value of the dependent variable.
        y_exact : Callable[..., npt.NDArray | float]
            The exact solution of the ODE.
        first_n : int
            The starting number of steps of the table (default: 2).
        n_rows : int
            The number of rows in the table (default: 8)

        Returns
        -------
        ConvergenceTableResult
            An object containing the results of the convergence table.

        Raises
        ------
        TypeError
            If y_start is not a number or a numpy array.
            If y_exact is not a callable object.
            If x_start or x_end is not a number.
            If first_n or n_rows is not a positive integer.
        ValueError
            If first_n or n_rows is not a positive integer.
        """
        if not isinstance(y_start, (int, float, np.ndarray)):
            raise TypeError("y_start must be a number or a numpy array")
        if not isinstance(y_exact, Callable):
            raise TypeError("y_exact must be a callable object")
        if not all(isinstance(x, (int, float)) for x in [x_start, x_end]):
            raise TypeError("x_start and x_end must be numbers (int or float)")
        if not all(isinstance(n, int) for n in [first_n, n_rows]):
            raise TypeError("first_n and n_rows must be integers")
        if not all(n > 0 for n in [first_n, n_rows]):
            raise ValueError("first_n and n_rows must be positive integers")

        y_start = np.atleast_1d(y_start)
        y_true_end = np.atleast_1d(y_exact(x_end))

        results = []
        last_error = 0.0

        for i in range(n_rows):
            n_steps = first_n * (2**i)
            h = (x_end - x_start) / n_steps

            x, y = x_start, y_start

            for _ in range(n_steps):
                y = y + h * self.function(x, y, *self.args, **self.kwargs)
                x += h

            error = np.linalg.norm(y - y_true_end)
            q = last_error / error if i > 0 else 0.0
            log2_q = np.log2(q) if q > 0 else 0.0

            results.append([n_steps, h, error, q, log2_q])
            last_error = error

        return ConvergenceTableResult(results)


# -----------------------------------------------------------------


if __name__ == "__main__":
    import matplotlib

    try:
        matplotlib.use("TkAgg")
        print("Backend do Matplotlib forçado para 'TkAgg'")
    except ImportError:
        print("Backend 'TkAgg' não encontrado, usando o padrão.")

    import matplotlib.pyplot as plt

    # --- Teste 1: EDO Escalar Simples ---
    # dy/dx = y, com y(0) = 1. A solução analítica é y(x) = e^x.

    print("Testando EDO Escalar: dy/dx = y")

    def f_scalar(x, y):
        return y

    def analytical_scalar(x):
        return np.exp(x)

    # Configurações
    x_start_scalar, x_end_scalar = 0, 4
    y_start_scalar = 1
    step_size_scalar = 0.1

    # Resolve com o método de Euler
    euler_scalar = EulerMethod(function=f_scalar, step_size=step_size_scalar)
    x_num, y_num = euler_scalar.solve(x_start_scalar, x_end_scalar, y_start_scalar)

    # Calcula a solução analítica para os mesmos pontos x
    y_analytical = analytical_scalar(x_num)

    # Cria o plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle("Teste da Classe EulerMethod", fontsize=16)

    ax1.plot(x_num, y_analytical, "b--", label="Analítica ($e^x$)")
    ax1.plot(x_num, y_num, "r-o", label=f"Euler (h={step_size_scalar})", markersize=3)
    ax1.set_title("Teste 1: EDO Escalar (dy/dx = y)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # --- Teste 2: Sistema de EDOs (Oscilador Harmônico Simples) ---
    # d²y/dt² = -y  =>  y(t) = cos(t) para y(0)=1, y'(0)=0
    # Sistema:
    # y1 = y  => dy1/dt = y2
    # y2 = y' => dy2/dt = -y1

    print("\nTestando Sistema de EDOs: Oscilador Harmônico Simples")

    def f_system(t, Y):
        y1, y2 = Y
        return np.array([y2, -y1])

    def analytical_system(t):
        # Retorna [y1(t), y2(t)]
        return np.array([np.cos(t), -np.sin(t)])

    # Configurações
    t_start_system, t_end_system = 0, 4 * np.pi
    y_start_system = np.array([1.0, 0.0])  # y(0)=1, y'(0)=0
    step_size_system = 0.1

    # Resolve com o método de Euler
    euler_system = EulerMethod(function=f_system, step_size=step_size_system)
    t_num, Y_num = euler_system.solve(t_start_system, t_end_system, y_start_system)

    # Y_num tem shape (n_steps, 2). A primeira coluna é y1, a segunda é y2.
    y1_num = Y_num[:, 0]
    y2_num = Y_num[:, 1]

    # Calcula a solução analítica para os mesmos pontos t
    Y_analytical = analytical_system(t_num)
    y1_analytical = Y_analytical[0, :]
    y2_analytical = Y_analytical[1, :]

    # Plota a posição (y1)
    ax2.plot(t_num, y1_analytical, "b--", label="Analítica (cos(t))")
    ax2.plot(t_num, y1_num, "r-", label=f"Euler (h={step_size_system})", markersize=3)
    ax2.set_title("Teste 2: Sistema de EDOs (Posição do Oscilador)")
    ax2.set_xlabel("Tempo (t)")
    ax2.set_ylabel("Posição (y)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Plota o espaço de fase (y2 vs y1)
    ax3.plot(y1_analytical, y2_analytical, "b--", label="Trajetória Analítica")
    ax3.plot(y1_num, y2_num, "r-", label="Trajetória de Euler")
    ax3.set_title("Teste 2: Espaço de Fase do Oscilador")
    ax3.set_xlabel("Posição (y1)")
    ax3.set_ylabel("Velocidade (y2)")
    ax3.legend()
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.set_aspect("equal", "box")

    # Nota sobre o erro do método de Euler para o oscilador:
    # É esperado que a amplitude da solução numérica aumente com o tempo.
    # Isso ocorre porque o método de Euler não conserva a energia do sistema.

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Teste 3: Tabela de Convergência ---
    # Vamos usar a EDO escalar do Teste 1 para gerar uma tabela de convergência.
    print("\n\n--- Teste 3: Tabela de Convergência para dy/dx = y ---")
    print("Gerando a tabela de convergência...")

    # O objeto euler_scalar já foi criado no Teste 1
    convergence_results = euler_scalar.convergence_table(
        x_start=x_start_scalar,
        x_end=x_end_scalar,
        y_start=y_start_scalar,
        y_exact=analytical_scalar,
        n_rows=8,  # Usando 6 linhas para um output mais conciso
    )

    print("Resultado da Tabela (formato padrão GitHub):")
    print(convergence_results)

    print("\nExportando a mesma tabela para o formato Typst:")
    print(convergence_results.export(format="typst", filename=None))

    print("\nExportando a mesma tabela para o formato LaTeX:")
    print(convergence_results.export(format="latex", filename=None))

    print("\nExportando a mesma tabela para o formato CSV:")
    print(convergence_results.export(format="csv", filename=None))

    print("\nExportando a mesma tabela para o formato JSON:")
    print(convergence_results.export(format="json", filename=None))

    plt.show()
