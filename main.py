from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pprint import pprint
from typing import Callable, Optional, Any

import numpy as np
import numpy.linalg
import scipy
import sympy
from scipy.constants import golden_ratio

from utils import tupled, derivative

Point = np.ndarray
LearningRateFunction = Callable[[float, float, Callable[[Point], float], Point, Point], float]
RFunction = Callable[[Point], float]


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


def calculate_next_point_classic(last_point: Point, gradient: np.ndarray, hessian: np.ndarray,
                                 **kwargs) -> Point:
    func = kwargs["function"]
    learning_rate_function = kwargs["lr_function"]
    grad = np.linalg.inv(hessian) @ np.transpose(gradient)
    return last_point - learning_rate_function(0, 1, func, last_point, grad) * grad


def calculate_next_point_linear_system(last_point: Point, gradient: np.ndarray, hessian: np.ndarray,
                                 **kwargs) -> Point:
    delta_x = np.linalg.solve(hessian, gradient)
    return last_point - delta_x


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


def gradient_descend(
    func: Callable[[Point], float],
    derivatives: list[Callable[[Point], Point]],
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
        grad = np.array([coord(path[-1]) for coord in derivatives])
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


def newton_descend(
    func,
    hessian: list[list[Callable[[Point], Point]]],
    derivatives: list[Callable[[Point], Point]],
    start: Point,
    learning_rate_function: LearningRateFunction,
    max_iter: int, *,
    stop_function_delta: Optional[float] = None,
    stop_point_delta: Optional[float] = None,
) -> NewtonOptimizationResult:
    # todo вынести копипасту
    if (stop_function_delta is not None) and stop_function_delta < 0:
        raise ValueError("Условие останова по значениям функции должно быть положительным")
    if (stop_point_delta is not None) and stop_point_delta < 0:
        raise ValueError("Условие останова по точкам должно быть положительным")
    path = [start]
    stop_reason = StopReason.ITERATIONS
    for _ in range(max_iter):
        grad = np.array([coord(path[-1]) for coord in derivatives])
        hessian_counter = lambda p: p(path[-1])
        calculated_hess = np.vectorize(hessian_counter)(hessian)
        new_point = calculate_next_point_func(path[-1], grad, calculated_hess, function=func, lr_function=learning_rate_function)
        # grad = np.linalg.inv(calculated_hess) @ np.transpose(grad)
        # new_point = path[-1] - learning_rate_function(0, 1, func, path[-1], grad) * grad
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

    return NewtonOptimizationResult(
        path=np.array(path),
        result=path[-1],
        iterations=len(path) - 1,
        success=stop_reason != StopReason.ITERATIONS,
        stop_reason=stop_reason
    )


fixed_nelder_mead = False


def add_sim():
    global fixed_nelder_mead

    if fixed_nelder_mead:
        return

    import inspect

    code = inspect.getsource(scipy.optimize._optimize._minimize_neldermead)
    marker = 'intermediate_result = OptimizeResult(x=sim[0], fun=fsim[0])'
    need_to_add = "intermediate_result['sim'] = sim"

    if need_to_add in code:
        return

    idx = code.index(marker) + len(marker)
    code = code[:idx] + '; ' + need_to_add + code[idx:]

    ns = {}
    exec(code, scipy.optimize._optimize.__dict__, ns)
    exec(inspect.getsource(scipy.optimize.minimize),
         scipy.optimize._minimize.__dict__ | ns, ns)
    scipy.optimize.minimize = ns['minimize']

    fixed_nelder_mead = True


def scipy_nelder_mead(f: RFunction, x0: Point, max_iterations=1000, verbose=True, *args, **kwargs):
    add_sim()

    path = []

    def callback(intermediate_result):
        path.append(intermediate_result['sim'])

    # return OptResult(res.x, res.nit, simplexes=path)

    res = scipy.optimize.minimize(
        f, x0, *args, method='Nelder-Mead',
        tol=1e-6, options=dict(disp=verbose, maxiter=max_iterations), callback=callback, **kwargs)

    return SimplexOptimizationResult(
        result=res.x, iterations=res.nit, simplexes=path,
        success=res.success,
        stop_reason=res.message
    )


def scipy_newton_cg(f: RFunction, x0: Point, max_iterations=1000, verbose=True, *args, **kwargs):
    res = scipy.optimize.minimize(
        f, x0, *args,
        method='Newton-CG', options=dict(disp=verbose, maxiter=max_iterations),
        **kwargs)
    return NewtonOptimizationResult(
        path=[], result=res.x, iterations=res.nit,
        stop_reason=res.message,
        success=res.success
    )


def calculate_hesse_matrix(func, vars) -> list[list[Callable[[Point], Point]]]:
    n = len(vars)
    hesse_matrix = []
    first_deratives = [func.diff(var) for var in vars]
    for i in range(n):
        raw = [tupled(sympy.lambdify(vars, first_deratives[i].diff(vars[j]), 'numpy')) for j in range(n)]
        hesse_matrix.append(raw)
    return hesse_matrix


def main():
    x, y = sympy.symbols('x y', real=True)
    f = lambda t: 0.001 * t[0] ** 2 + t[1] ** 2
    f_sp = f([x, y])
    # path = gradient_descend(
    #     f,
    #     derivatives=derivative(f_sp, [x, y]),
    #     start=np.array([10., 10.]),
    #     learning_rate_function=golden_ratio_method(1e-10),
    #     max_iter=1000
    # )
    # pprint(path)
    path = Newton_descend(tupled(sympy.lambdify([x, y], f_sp, 'numpy')),
                          hessian=calculate_hesse_matrix(f_sp, [x, y]),
                          calculate_next_point_func=calculate_next_point_linear_system,
                          derivatives=derivative(f_sp, [x, y]),
                          start=np.array([10., 10.]),
                          learning_rate_function=golden_ratio_method(1e-10),
                          stop_function_delta=1e-8,
                          max_iter=1000)
    pprint(path)
    scipy_nelder_mead(f, np.array([10., 10.]))
    # path = newton_descend(tupled(sympy.lambdify([x, y], f_sp, 'numpy')),
    #                       hessian=calculate_hesse_matrix(f_sp, [x, y]),
    #                       derivatives=derivative(f_sp, [x, y]),
    #                       start=np.array([10., 10.]),
    #                       learning_rate_function=golden_ratio_method(1e-10),
    #                       stop_function_delta=1e-8,
    #                       max_iter=1000)
    # pprint(path)
    # scipy_nelder_mead(f, np.array([10., 10.]))

    scipy_newton_cg(f, np.array([10, 10]))


if __name__ == '__main__':
    main()
