from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
import sympy

Point = np.ndarray
Vector = np.ndarray
LearningRateFunction = Callable[[float, float, Callable[[Point], float], Point, Point], float]
LearningRateFactoryFunction = Callable[[float], LearningRateFunction]
RFunction = Callable[[Point], float]

NewtonNextPointFunction = Callable[[RFunction, LearningRateFunction, Point, np.ndarray, np.ndarray], Point]


class StopReason(Enum):
    ITERATIONS: str = "iterations"
    FUNCTION_DELTA: str = "function_delta"
    POINT_DELTA: str = "point_delta"
    GRADIENT_NORM: str = "gradient_norm"
    EPOCH: str = "epoch_number"
    NAN: str = "nan"


@dataclass
class OptimizationResult:
    result: Point
    iterations: int
    stop_reason: StopReason
    success: bool


@dataclass
class SimplexOptimizationResult(OptimizationResult):
    simplexes: list[Point]


@dataclass
class DescentOptimizationResult(OptimizationResult):
    path: np.ndarray | list[Point]


@dataclass
class GradientOptimizationResult(DescentOptimizationResult):
    pass


@dataclass
class StochasticGradientOptimizationResult(GradientOptimizationResult):
    loss: float


@dataclass
class NewtonOptimizationResult(DescentOptimizationResult):
    pass


MinimizeSympyFunction = Callable[[sympy.Expr, Point, LearningRateFunction, int, float, float], OptimizationResult]
NewtonFunction = Callable[[RFunction, NewtonNextPointFunction, Callable[[Point], np.ndarray],
                           Callable[[Point], np.ndarray], Point, LearningRateFunction, int, float, float], GradientOptimizationResult]
