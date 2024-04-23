from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pprint import pprint
from typing import Callable, Optional, Any

import numpy as np
import numpy.linalg
import scipy
import sympy

import funcs as fs
from utils import tupled, derivative
from typing_local import *



def main():
    x, y = sympy.symbols('x y', real=True)
    # f = lambda t: (1 - t[0]) ** 2 + 10 * (t[1] - t[0] ** 2) ** 2
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
    # path = newton_descend(tupled(sympy.lambdify([x, y], f_sp, 'numpy')),
    #                       hessian=calculate_hesse_matrix(f_sp, [x, y]),
    #                       calculate_next_point_func=calculate_next_point_linear_system,
    #                       derivatives=derivative(f_sp, [x, y]),
    #                       start=np.array([10., 10.]),
    #                       learning_rate_function=golden_ratio_method(1e-10),
    #                       stop_function_delta=1e-8,
    #                       max_iter=1000)
    # scipy_nelder_mead(f, np.array([10., 10.]))
    print("Newton descend")
    pprint(newton_descend(f,
                          calculate_next_point_func=calculate_next_point_classic,
                          hess=fs.hessian(f_sp),
                          # hessian=calculate_hesse_matrix(f_sp, [x, y]),
                          jac=fs.jacobi(f_sp),
                          start=np.array([3.5, 12.5]),
                          learning_rate_function=golden_ratio_method(1e-10),
                          stop_function_delta=1e-10,
                          max_iter=1000))
    print()
    print("Scipy nelder mead")
    (scipy_nelder_mead(f, np.array([3.5, 12.5])))

    print()
    print("Scipy newton cg")
    pprint(scipy_newton_cg(f, jac=fs.jacobi(f_sp), hessian=fs.hessian(f_sp), x0=np.array([3.5, 12.5])))


if __name__ == '__main__':
    main()
