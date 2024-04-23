from functools import wraps

from typing_local import Callable, Point

import numpy as np
import sympy as sp

from utils import tupled


def hessian(f: sp.Expr) -> Callable[[Point], np.ndarray]:
    hess = sp.hessian(f, list(f.free_symbols))
    return lambda x: np.array(lambdify(hess)(x))


def lambdify(f: sp.Expr) -> Callable[[Point], float]:
    def lambdified(x):
        lam = f.evalf(subs={k: v for k, v in zip(f.free_symbols, x)})

        if lam.is_Matrix:
            return np.array(lam).astype('float64')

        return np.float64(lam)

    return lambdified


def jacobi(f: sp.Expr) -> Callable[[Point], np.ndarray]:
    jac = sp.Matrix([f]).jacobian(list(f.free_symbols)).transpose()
    return lambda x: np.array(lambdify(jac)(x)).squeeze()
