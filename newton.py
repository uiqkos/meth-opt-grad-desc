from typing import Optional
import lbfgs

import scipy

from typing_local import *


def calculate_next_point_classic(
    f: RFunction,
    lr: LearningRateFunction,
    last_point: Point,
    gradient: np.ndarray,
    hessian: np.ndarray
) -> Point:
    grad = np.linalg.inv(hessian) @ np.transpose(gradient)
    return last_point - lr(0, 1, f, last_point, grad) * grad


def calculate_next_point_linear_system(
    f: RFunction,
    lr: LearningRateFunction,
    last_point: Point,
    gradient: np.ndarray,
    hessian: np.ndarray
) -> Point:
    delta_x = np.linalg.solve(hessian, gradient)
    return last_point - delta_x


def newton_descend(
    func,
    calculate_next_point_func: NewtonNextPointFunction,
    hess: Callable[[Point], np.ndarray],
    jac: Callable[[Point], np.ndarray],
    start: Point,
    learning_rate_function: LearningRateFunction,
    max_iter: int,
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
        new_point = calculate_next_point_func(func, learning_rate_function, path[-1], jac(path[-1]), hess(path[-1]))
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


def scipy_newton_cg(f: RFunction, jac, hessian, x0: Point, max_iterations=1000, verbose=True):
    res = scipy.optimize.minimize(
        f, x0, jac=jac,
        method='Newton-CG', options={'xtol': 1e-8,
                                     'disp': False,
                                     'return_all': True,
                                     'maxiter': max_iterations}
    )
    return NewtonOptimizationResult(
        path=[x0] + res.allvecs, result=res.x, iterations=res.nit,
        stop_reason=res.message,
        success=res.success
    )


def pylbfgs_lbfgs(
    f: RFunction,
    jac: Callable[[Point], np.ndarray],
    x0: Point,
):
    l = lbfgs.LBFGS()

    def fg(x, g):
        g[:] = jac(x)
        return f(x)

    path = [x0]
    iters = [0]

    # callable(x, g, fx, xnorm, gnorm, step, k, num_eval, *args)
    def callback(x, g, fx, xnorm, gnorm, step, k, num_eval, *args):
        path.append(x)
        iters[0] = k

    res = l.minimize(fg, x0, callback)

    return DescentOptimizationResult(
        res, iters[0],
        stop_reason=StopReason.ITERATIONS,
        success=True,
        path=path + [res]
    )
