from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pprint import pprint
from typing import Callable, Optional, Any
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg
import scipy
import sympy

import funcs as fs
from grad_desc import *
from nelder import scipy_nelder_mead
from newton import newton_descend, scipy_newton_cg
from utils import tupled, derivative
from typing_local import *


def plot_points(x, y, koef):
    plt.scatter(x, y)
    test_x = np.linspace(-10, 10, 100)
    test_y = np.array([koef[0] * e for e in test_x])
    plt.plot(test_x, test_y, linewidth=2, color="red")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of Points')
    plt.show()


def generate_linear_data(dim: int = 1, size: int = 500):
    # X = np.array([np.random. * 100 for _ in range(size)])
    x = np.random.randn(size) * 10
    coefs = np.random.random_sample() * 10 + np.random.randn()
    # b = np.random.randn(size) * 12
    # print(coefs)
    # print(coefs @ X)
    # y = X @ coefs + torch.randn(size, 1) * 0.5
    y = coefs * x
    plot_points(x, y, [0])
    return x, y


def main():
    x, y = sympy.symbols('x y', real=True)

    X, Y = generate_linear_data()
    X = np.array([np.array([number]) for number in X])
    f = lambda vect, ind: (vect @ X[ind] - Y[ind]) ** 2
    f_derivative = lambda vect, ind: 2 * ((vect @ X[ind]) - Y[ind]) * X[ind]
    result = stochastic_gradient_descent(objective_func_term=f,
                                         objective_func_term_derivative=f_derivative,
                                         data_size=X.size,
                                         batch_size=200,
                                         learning_rate_function=piecewise_learning_rate_scheduling(np.array([1, 3, 4]),
                                                                                                   np.array(
                                                                                                       [0.01, 0.005,
                                                                                                        0.0001,
                                                                                                        0.0001])),
                                         # learning_rate_function=polynomial_learning_rate_scheduling(0.01, 1/2, 1),
                                         max_iter=2000,
                                         start=np.array([0]))

    plot_points(X, Y, result.result)
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
    # # pprint(newton_descend(f,
    #                       calculate_next_point_func=calculate_next_point_classic,
    #                       hess=fs.hessian(f_sp),
    #                       # hessian=calculate_hesse_matrix(f_sp, [x, y]),
    #                       jac=fs.jacobi(f_sp),
    #                       start=np.array([3.5, 12.5]),
    #                       learning_rate_function=golden_ratio_method(1e-10),
    #                       stop_function_delta=1e-10,
    #                       max_iter=1000))
    # print()
    # print("Scipy nelder mead")
    # (scipy_nelder_mead(f, np.array([3.5, 12.5])))
    #
    # print()
    # print("Scipy newton cg")
    # pprint(scipy_newton_cg(f, jac=fs.jacobi(f_sp), hessian=fs.hessian(f_sp), x0=np.array([3.5, 12.5])))


if __name__ == '__main__':
    main()
