from functools import wraps

import sympy

from typing_local import Callable, Point

import numpy as np
import sympy as sp

from utils import tupled


def hessian(f: sp.Expr) -> Callable[[Point], np.ndarray]:
    hess = sp.hessian(f, list(f.free_symbols))
    return lambda x: np.array(lambdify(hess)(x))


def lambdify(f: sp.Expr) -> Callable[[Point], float]:
    def lambdified(x):
        lam = f.evalf(subs={k: v for k, v in zip(sorted(list(f.free_symbols), key=str), x)})

        if lam.is_Matrix:
            return np.array(lam).astype('float64')

        return np.float64(lam)

    return lambdified


def jacobi(f: sp.Expr) -> Callable[[Point], np.ndarray]:
    jac = sp.Matrix([f]).jacobian(sorted(list(f.free_symbols), key=str)).transpose()
    return lambda x: np.array(lambdify(jac)(x)).squeeze()


def from_sympy_to_plotable_func(f):
    return tupled(np.vectorize(sympy.lambdify(sympy.symbols('x y'), f, 'numpy')))
