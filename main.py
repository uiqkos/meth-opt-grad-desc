from functools import wraps
from typing import Callable

import numpy as np
import sympy
from scipy.optimize import minimize

Point = np.ndarray


def gradient_descend(
    func: Callable[[Point], float],
    derivatives: list[Callable[[Point], Point]],
    start: Point,
    learning_rate: float,
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
        new_point = path[-1] - learning_rate * grad
        path.append(new_point)

        if (stop_function_delta is not None) and abs(func(path[-1]) - func(path[-2])) < stop_function_delta:
            break

        if (stop_point_delta is not None) and np.linalg.norm(path[-1] - path[-2]) < stop_point_delta:
            break

    return path


def scipi_nelder_mead(f, x0):
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
    path = gradient_descend(f, derivative(f_sp, [x, y]), np.array([10, 10]), 0.1, 10 ** 100,
                            stop_function_delta=1e-15)

    scipi_nelder_mead(f, np.array([10., 10.]))
