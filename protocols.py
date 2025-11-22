from typing import Any, Callable, Optional, Protocol, runtime_checkable
import numpy.typing as npt


@runtime_checkable
class LinearSolver(Protocol):
    """
    Protocol defining the interface for linear equation solvers.
    Any class implementing this solve method is a valid LinearSolver.
    """

    def solve(
        self,
        A: npt.NDArray,
        b: npt.NDArray,
        x0: Optional[npt.NDArray] = None,
    ) -> npt.NDArray: ...


@runtime_checkable
class RootSolver(Protocol):
    """
    Protocol defining the interface for root-finding algorithms.
    """

    def solve(
        self,
        function: Callable[..., npt.NDArray],
        x0: npt.NDArray,
        *args: Any,
        **kwargs: Any
    ) -> npt.NDArray: ...


@runtime_checkable
class ExplicitStepper(Protocol):
    """
    Protocol defining the logic for a single explicit step.
    Used as a predictor strategy or by explicit solvers.
    """

    def step(
        self,
        function: Callable[..., npt.NDArray],
        x: float,
        y: npt.NDArray,
        step_size: float,
        *args: Any,
        **kwargs: Any
    ) -> npt.NDArray: ...
