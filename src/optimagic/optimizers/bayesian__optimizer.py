"""Implement Bayesian optimization algorithms."""

from dataclasses import dataclass

import numpy as np
from bayes_opt import BayesianOptimization, acquisition
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_BAYESOPT_INSTALLED
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    PositiveInt,
)


@mark.minimizer(
    name="bayes_opt",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_BAYESOPT_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=True,
    disable_history=False,
)
@dataclass(frozen=True)
class BayesOpt(Algorithm):
    """Bayesian Optimization wrapper using bayes_opt package.

    Args:
        init_points: Number of initial random exploration points
        n_iter: Number of optimization iterations
        kappa: Exploration-exploitation trade-off parameter for UCB
        xi: Exploration-exploitation trade-off parameter for EI and PI
        verbose: Verbosity level (0-3)
        acquisition_function: Type of acquisition function ("ucb", "ei", or "poi")
        random_state: Random seed for reproducibility
        allow_duplicate_points: Whether to allow duplicate evaluation points
        enable_sdr: Enable Sequential Domain Reduction
        sdr_gamma_osc: Oscillation parameter for SDR
        sdr_gamma_pan: Panning parameter for SDR
        sdr_eta: Zooming parameter for SDR
        sdr_minimum_window: Minimum window size for SDR

    """

    init_points: PositiveInt = 5
    n_iter: PositiveInt = 50
    verbose: int = 2
    kappa: float = 2.576
    xi: float = 0.01
    exploration_decay: float | None = None
    exploration_decay_delay: int | None = None
    random_state: int | None = None
    acquisition_function: str = "ucb"
    allow_duplicate_points: bool = False
    enable_sdr: bool = False
    sdr_gamma_osc: float = 0.7
    sdr_gamma_pan: float = 1.0
    sdr_eta: float = 0.9
    sdr_minimum_window: float = 0.0

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not (
            problem.bounds.lower is not None
            and problem.bounds.upper is not None
            and np.all(np.isfinite(problem.bounds.lower))
            and np.all(np.isfinite(problem.bounds.upper))
        ):
            raise ValueError(
                "Bayesian optimization requires finite bounds for all parameters. "
                "Bounds cannot be None or infinite."
            )

        # Convert bounds to dictionary format required by BayesianOptimization
        pbounds = {
            f"param{i}": (lower, upper)
            for i, (lower, upper) in enumerate(
                zip(problem.bounds.lower, problem.bounds.upper, strict=True)
            )
        }

        common_kwargs = {
            "exploration_decay": self.exploration_decay,
            "exploration_decay_delay": self.exploration_decay_delay,
            "random_state": self.random_state,
        }

        if self.acquisition_function == "ucb":
            acq = acquisition.UpperConfidenceBound(kappa=self.kappa, **common_kwargs)
        elif self.acquisition_function == "poi":
            acq = acquisition.ProbabilityOfImprovement(xi=self.xi, **common_kwargs)
        elif self.acquisition_function == "ei":
            acq = acquisition.ExpectedImprovement(xi=self.xi, **common_kwargs)
        else:
            raise ValueError(
                f"Invalid acquisition_function: {self.acquisition_function}. "
                "Must be one of: 'ucb', 'poi', 'ei'"
            )

        # Process constraints
        constraint = None
        # constraint = self._process_constraints(problem.nonlinear_constraints)

        def objective(**kwargs: dict[str, float]) -> float:
            x = np.array([kwargs[f"param{i}"] for i in range(len(x0))])
            return -float(
                problem.fun(x)
            )  # Negate to convert minimization to maximization

        # Configure domain reduction
        bounds_transformer = None
        if self.enable_sdr:
            from bayes_opt import SequentialDomainReductionTransformer

            bounds_transformer = SequentialDomainReductionTransformer(
                gamma_osc=self.sdr_gamma_osc,
                gamma_pan=self.sdr_gamma_pan,
                eta=self.sdr_eta,
                minimum_window=self.sdr_minimum_window,
            )

        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            constraint=constraint,
            acquisition_function=acq,
            random_state=self.random_state,
            verbose=self.verbose,
            allow_duplicate_points=self.allow_duplicate_points,
            bounds_transformer=bounds_transformer,
        )

        optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter,
        )

        best_params = optimizer.max["params"]
        best_x = np.array([best_params[f"param{i}"] for i in range(len(x0))])
        best_y = -optimizer.max["target"]  # Un-negate the result

        return InternalOptimizeResult(
            x=best_x,
            fun=best_y,
            success=True,
            n_iterations=self.init_points + self.n_iter,
            n_fun_evals=self.init_points + self.n_iter,
        )

    # def _process_constraints(self, constraints: Any) -> None:
    #     TODO: Implement constraint processing
    #     return None
