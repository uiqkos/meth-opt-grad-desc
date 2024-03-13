from functools import wraps
from typing import Callable

import numpy as np
import sympy
from scipy.optimize import minimize

Point = np.ndarray


def constant_rate(left: float, right: float, function: Callable[[float], float], stop_delta: float) -> float:
    return stop_delta


def dichotomy_method(left: float, right: float, function: Callable[[float], float], stop_delta: float) -> float:
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


def gradient_descend(
    func: Callable[[Point], float],
    derivatives: list[Callable[[Point], Point]],
    start: Point,
    learning_rate_function: Callable[[float, float, Callable[[float], float], float], float],
    max_iter: int, *,
    stop_function_delta: float | None = None,
    stop_point_delta: float | None = None,
) -> list[Point]:
    if (stop_function_delta is not None) and stop_function_delta < 0:
        raise ValueError("Условие останова по значениям функции должно быть положительным")
    if (stop_point_delta is not None) and stop_point_delta < 0:
        raise ValueError("Условие останова по точкам должно быть положительным")
    path = [start]

    for _ in range(max_iter):
        grad = np.array([coord(path[-1]) for coord in derivatives])
        new_point = path[-1] - learning_rate_function(0, 0.01, lambda l: func(path[-1] - l * grad), 1e-10) * grad
        path.append(new_point)

        if (stop_function_delta is not None) and abs(func(path[-1]) - func(path[-2])) < stop_function_delta:
            break

        if (stop_point_delta is not None) and np.linalg.norm(path[-1] - path[-2]) < stop_point_delta:
            break

    return path


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
    path = gradient_descend(f, derivative(f_sp, [x, y]), np.array([10, 10]), constant_rate, 10 ** 100,
                            stop_function_delta=1e-15)
    print(path[-1])
    scipy_nelder_mead(f, np.array([10., 10.]))
