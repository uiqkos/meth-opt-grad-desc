from typing import Optional

from scipy.constants import golden_ratio

from typing_local import *


def constant_rate(lr: float) -> LearningRateFunction:
    def rate_function(*_):
        return lr

    rate_function.__name__ = "constant_rate"

    return rate_function


def dichotomy_method(stop_delta: float) -> LearningRateFunction:
    def rate_function(left: float, right: float,
                      func: Callable[[Point], float], last_point: Point, grad: Point) -> float:
        function = lambda lr: func(last_point - lr * grad)
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

    rate_function.__name__ = "dichotomy_method"

    return rate_function


def golden_ratio_method(stop_delta: float) -> LearningRateFunction:
    def rate_function(left: float, right: float,
                      func: Callable[[Point], float], last_point: Point, grad: Point) -> float:
        function = lambda lr: func(last_point - lr * grad)
        x1 = left + (right - left) / golden_ratio ** 2
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

    rate_function.__name__ = "golden_ratio_method"

    return rate_function


def gradient_descend(
    func: Callable[[Point], float],
    jac: Callable[[Point], np.ndarray],
    start: Point,
    learning_rate_function: LearningRateFunction,
    max_iter: int, *,
    stop_function_delta: Optional[float] = None,
    stop_point_delta: Optional[float] = None,
) -> GradientOptimizationResult:
    if (stop_function_delta is not None) and stop_function_delta < 0:
        raise ValueError("Условие останова по значениям функции должно быть положительным")
    if (stop_point_delta is not None) and stop_point_delta < 0:
        raise ValueError("Условие останова по точкам должно быть положительным")

    path = [start]
    stop_reason = StopReason.ITERATIONS
    for _ in range(max_iter):
        grad = jac(path[-1])
        new_point = path[-1] - learning_rate_function(0, 0.1, func, path[-1], grad) * grad
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

    return GradientOptimizationResult(
        result=path[-1],
        iterations=len(path) - 1,
        stop_reason=stop_reason,
        success=stop_reason != StopReason.ITERATIONS,
        path=np.array(path),
    )
