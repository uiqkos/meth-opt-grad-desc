from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pprint import pprint
from typing import Callable, Optional, Any


import numpy as np
import sympy
from scipy.optimize import minimize
from scipy.constants import golden_ratio

Point = np.ndarray
LearningRateFunction = Callable[[float, float, Callable[[float], float]], float]


def constant_rate(lr: float) -> LearningRateFunction:
    def rate_function(_, __, ___):
        return lr

    return rate_function


def dichotomy_method(stop_delta: float) -> LearningRateFunction:
    def rate_function(left: float, right: float, function: Callable[[float], float]) -> float:
        middle = (left + right) / 2
        middle_val = function(middle)
        while (right - left) > stop_delta:
            x1 = (left + middle) / 2
            if (x1_val := function(x1)) <= middle_val:
                right = middle
                middle = x1
                middle_val = x1_val
            else:
                x2 = (middle + right) / 2
                if (x2_val := function(x2)) < middle_val:
                    left = middle
                    middle = x2
                    middle_val = x2_val
                    continue
                left = x1
                right = x2
        return middle

    return rate_function


def golden_ratio_method(stop_delta: float) -> LearningRateFunction:
    def rate_function(left: float, right: float, function: Callable[[float], float]) -> float:
        x1 = left + (right - left) / golden_ratio**2
        x2 = left + (right - left) / golden_ratio
        x1_val = function(x1)
        x2_val = function(x2)
        while (right - left) > stop_delta:
            if x1_val <= x2_val:
                right = x2
                x2 = x1
                x1 = left + (right - left) / golden_ratio ** 2
                x1_val = function(x1)
            else:
                left = x1
                x1 = x2
                x2 = left + (right - left) / golden_ratio
                x2_val = function(x2)
        return (left + right) / 2

    return rate_function


class StopReason(Enum):
    ITERATIONS: str = "iterations"
    FUNCTION_DELTA: str = "function_delta"
    POINT_DELTA: str = "point_delta"
    NAN: str = "nan"


@dataclass
class GradientDescendResult:
    path: np.ndarray
    result: Point
    iterations: int
    stop_reason: StopReason


def gradient_descend(
    func: Callable[[Point], float],
    derivatives: list[Callable[[Point], Point]],
    start: Point,
    learning_rate_function: LearningRateFunction,
    max_iter: int, *,
    stop_function_delta: Optional[float] = None,
    stop_point_delta: Optional[float] = None,
) -> GradientDescendResult:
    if (stop_function_delta is not None) and stop_function_delta < 0:
        raise ValueError("Условие останова по значениям функции должно быть положительным")
    if (stop_point_delta is not None) and stop_point_delta < 0:
        raise ValueError("Условие останова по точкам должно быть положительным")
    path = [start]
    stop_reason = StopReason.ITERATIONS

    for _ in range(max_iter):
        grad = np.array([coord(path[-1]) for coord in derivatives])
        new_point = path[-1] - learning_rate_function(0, 0.01, lambda l: func(path[-1] - l * grad)) * grad
        path.append(new_point)

        if np.isnan(new_point).any():
            stop_reason = StopReason.NAN
            break

        if (stop_function_delta is not None) and abs(func(path[-1]) - func(path[-2])) < stop_function_delta:
            stop_reason = StopReason.FUNCTION_DELTA
            break

        if (stop_point_delta is not None) and np.linalg.norm(path[-1] - path[-2]) < stop_point_delta:
            stop_reason = StopReason.POINT_DELTA
            break

    return GradientDescendResult(
        path=np.array(path),
        result=path[-1],
        iterations=len(path) - 1,
        stop_reason=stop_reason
    )


def scipy_nelder_mead(f, x0):
    res = minimize(f, x0, method='Nelder-Mead', tol=1e-6, options=dict(disp=True))
    print(res)


def tupled(f):
    @wraps(f)
    def wrapper(args):
        return f(*args)

    return wrapper


def derivative(f, vars):
    diffs = [tupled(sympy.lambdify(vars, f.diff(var), 'numpy')) for var in vars]
    return diffs


if __name__ == '__main__':
    x, y = sympy.symbols('x y', real=True)
    f = lambda t: t[0] ** 2 + t[1] ** 2
    f_sp = f([x, y])
    path = gradient_descend(
        f,
        derivatives=derivative(f_sp, [x, y]),
        start=np.array([10., 10.]),
        learning_rate_function=golden_ratio_method(1e-10),
        max_iter=1000,
        # stop_function_delta=1e-10,
        # stop_point_delta=1e-10
    )
    pprint(path)
    scipy_nelder_mead(f, np.array([10., 10.]))
