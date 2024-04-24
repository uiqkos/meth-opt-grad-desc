import numpy as np
import pandas as pd
import sympy
import sympy as sp
import funcs as fs
from grad_desc import *
from newton import *
from nelder import *

from newton import calculate_next_point_classic
import utils as u
from typing_local import *


def gradient_descent_(
    func: sp.Expr,
    start_point: Point,
    learning_rate_function: LearningRateFunction,
    max_iter: int = 1000,
    stop_point_delta: float = 1e-10,
    stop_function_delta: float = 1e-10
):
    return gradient_descend(
        fs.lambdify(func),
        jac=fs.jacobi(func),
        start=start_point,
        learning_rate_function=learning_rate_function,
        max_iter=max_iter,
        stop_point_delta=stop_point_delta,
        stop_function_delta=stop_function_delta
    )


def scipy_nelder_mead_(
    func: sp.Expr,
    start_point: Point,
    learning_rate_function: LearningRateFunction,
    max_iter: int = 1000,
    stop_point_delta: float = 1e-10,
    stop_function_delta: float = 1e-10
):
    return scipy_nelder_mead(
        fs.lambdify(func),
        x0=start_point,
        max_iterations=max_iter,
    )


def newton_descent_classic_next_point(
    func: sp.Expr,
    start_point: Point,
    learning_rate_function: LearningRateFunction,
    max_iter: int = 1000,
    stop_point_delta: float = 1e-10,
    stop_function_delta: float = 1e-10
):
    return newton_descend(
        fs.lambdify(func),
        calculate_next_point_func=calculate_next_point_classic,
        learning_rate_function=learning_rate_function,
        jac=fs.jacobi(func),
        hess=fs.hessian(func),
        start=start_point,
        max_iter=max_iter,
        stop_point_delta=stop_point_delta,
        stop_function_delta=stop_function_delta
    )


def newton_descent_linalg_next_point(
    func: sp.Expr,
    start_point: Point,
    learning_rate_function: LearningRateFunction,
    max_iter: int = 1000,
    stop_point_delta: float = 1e-10,
    stop_function_delta: float = 1e-10
):
    return newton_descend(
        fs.lambdify(func),
        calculate_next_point_func=calculate_next_point_linear_system,
        learning_rate_function=learning_rate_function,
        jac=fs.jacobi(func),
        hess=fs.hessian(func),
        start=start_point,
        max_iter=max_iter,
        stop_point_delta=stop_point_delta,
        stop_function_delta=stop_function_delta
    )


def scipy_newton_(
    func: sp.Expr,
    start_point: Point,
    learning_rate_function: LearningRateFunction,
    max_iter: int = 1000,
    stop_point_delta: float = 1e-10,
    stop_function_delta: float = 1e-10
):
    return scipy_newton_cg(
        fs.lambdify(func),
        jac=fs.jacobi(func),
        hessian=fs.hessian(func),
        x0=start_point,
        max_iterations=max_iter,
    )


methods = [
    gradient_descent_,
    scipy_nelder_mead_,
    newton_descent_classic_next_point,
    newton_descent_linalg_next_point,
    scipy_newton_,
]
