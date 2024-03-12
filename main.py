from functools import wraps
from typing import Callable

import numpy as np
import scipy
import sympy

Point = np.ndarray


def gradient_descend(
    func: Callable[[Point], float],
    derivative: list[Callable[[float], Point]],
    start: Point,
    learning_rate: float,
    stop: float = 0.001,
    max_iter: int = 1000,
) -> list[Point]:
    path = [start]

    for _ in range(max_iter):
        # grad = np.array([d(*path[-1]) for d in derivative])
        grad = np.array([coord(*path[-1]) for coord in derivative])
        new_point = path[-1] - learning_rate * grad
        path.append(new_point)

        if abs(func(*path[-1]) - func(*path[-2])) < stop:
            break

    return path


def simplify(f, vars):
    @wraps(f)
    def wrapper(*args):
        return f.subs(tuple(zip(vars, args))).evalf()

    return wrapper


def derivative(f, vars):
    diffs = [simplify(f.diff(var), vars) for var in vars]
    return diffs


if __name__ == '__main__':
    x, y = sympy.symbols('x y', real=True)
    f_sp = x ** 2 - 1.34 * y ** 2 - 2 * y + 0.14
    f = simplify(f_sp, [x, y])
    path = gradient_descend(f, derivative(f_sp, [x, y]), np.array([0, 0]), 0.1, 0.001, 1000)

    print(path)
    print(path[-1])

#%%

#%%
