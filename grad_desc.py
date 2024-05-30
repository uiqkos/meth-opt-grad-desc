from typing import Optional

import numpy as np
from scipy.constants import golden_ratio

from typing_local import *
from bisect import bisect_left


def constant_learning_rate(start_learning_rate: float):
    def learning_rate_function(*_):
        return start_learning_rate
    learning_rate_function.__name__ = "constant_learning_rate"
    return learning_rate_function


def exponential_learning_rate_scheduling(start_learning_rate: float, _lambda: float):
    def learning_rate_function(epoch_number: int):
        return start_learning_rate * np.e ** (-_lambda * epoch_number)
    learning_rate_function.__name__ = "exponential_scheduling"
    return learning_rate_function


def polynomial_learning_rate_scheduling(start_learning_rate: float, alpha: float,
                                        beta: float):

    def learning_rate_function(epoch_number: int):
        return start_learning_rate * (beta * epoch_number + 1) ** -alpha

    learning_rate_function.__name__ = "polynomial_scheduling"
    return learning_rate_function


def piecewise_learning_rate_scheduling(borders: np.array, values: np.array):
    def learning_rate_function(epoch_number: int):
        return values[bisect_left(borders, epoch_number)]

    learning_rate_function.__name__ = "piecewise_scheduling"
    return learning_rate_function


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


def gradient_and_newton(
        func,
        newton_minimize_function: NewtonFunction,
        calculate_next_point_func: NewtonNextPointFunction,
        hess: Callable[[Point], np.ndarray],
        jac: Callable[[Point], np.ndarray],
        start: Point,
        learning_rate_function: LearningRateFunction,
        max_iter: int,
        stop_function_delta: Optional[float] = None,
        stop_point_delta: Optional[float] = None, ):
    grad_res = gradient_descend(func, jac, start, learning_rate_function, max_iter,
                                stop_function_delta=stop_function_delta,
                                stop_point_delta=stop_point_delta)
    newton_res = newton_minimize_function(func, calculate_next_point_func, hess, jac, grad_res.result,
                                          learning_rate_function,
                                          max_iter, stop_function_delta, stop_point_delta)
    return GradientOptimizationResult(
        result=newton_res.result,
        iterations=grad_res.iterations + newton_res.iterations,
        stop_reason=newton_res.stop_reason,
        success=newton_res.success,
        path=np.concatenate((grad_res.path, newton_res.path[1:]))
    )


def stochastic_gradient_descent(
        objective_func_term: Callable[[Vector, int], float],
        objective_func_term_derivative: Callable[[Vector, int], Vector],
        data_size: int,
        batch_size: int,
        start: Vector,
        learning_rate_function: Callable[[int], float],
        max_iter: int, *,
        stop_grad_norm: Optional[float] = None,
        stop_epoch_number: Optional[int] = None,
        calculate_loss: Optional[bool] = False) -> StochasticGradientOptimizationResult:
    dim = start.size
    if (stop_grad_norm is not None) and stop_grad_norm < 0:
        raise ValueError("Условие останова по норме градиента должно быть положительным")
    if (stop_epoch_number is not None) and stop_epoch_number <= 0:
        raise ValueError("Условие останова по количеству эпох должно быть положительным")
    if batch_size <= 0:
        raise ValueError("Размер батча должен быть положительным")
    if batch_size > data_size:
        raise ValueError("Размер батча должен быть не больше размера данных")
    loss_calculator = lambda vect: sum((objective_func_term(vect, i) for i in range(data_size))) / data_size
    path = [start]
    learning_rate = learning_rate_function(0)
    stop_reason = StopReason.ITERATIONS
    indexes = np.arange(data_size)
    np.random.shuffle(indexes)
    epoch = 0
    loss = None
    if calculate_loss:
        loss = loss_calculator(start)
    start_index = 0
    for _ in range(max_iter):
        grad = np.zeros(dim)
        for i in indexes[start_index:start_index + batch_size]:
            grad += objective_func_term_derivative(path[-1], i)
        grad /= batch_size
        new_point = path[-1] - learning_rate * grad
        if calculate_loss:
            loss = loss_calculator(new_point)
        path.append(new_point)
        start_index += batch_size
        if start_index + batch_size >= data_size:
            epoch += 1
            learning_rate = learning_rate_function(epoch)
            np.random.shuffle(indexes)
            start_index = 0
        if np.isnan(new_point).any():
            stop_reason = StopReason.NAN
            break
        if (stop_grad_norm is not None) and np.linalg.norm(grad) < stop_grad_norm:
            stop_reason = StopReason.GRADIENT_NORM
            break
        if (stop_epoch_number is not None) and epoch >= stop_epoch_number:
            stop_reason = StopReason.EPOCH
            break
    return StochasticGradientOptimizationResult(
        result=path[-1],
        iterations=len(path) - 1,
        stop_reason=stop_reason,
        success=stop_reason != StopReason.ITERATIONS,
        path=np.array(path),
        loss=loss
    )
