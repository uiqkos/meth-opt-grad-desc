from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

Point = np.ndarray
LearningRateFunction = Callable[[float, float, Callable[[Point], float], Point, Point], float]
RFunction = Callable[[Point], float]

NewtonNextPointFunction = Callable[[RFunction, LearningRateFunction, Point, np.ndarray, np.ndarray], Point]


class StopReason(Enum):
    ITERATIONS: str = "iterations"
    FUNCTION_DELTA: str = "function_delta"
    POINT_DELTA: str = "point_delta"
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
class NewtonOptimizationResult(DescentOptimizationResult):
    pass
