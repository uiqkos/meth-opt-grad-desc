import scipy

from typing_local import *

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

    res = scipy.optimize.minimize(
        f, x0, *args, method='Nelder-Mead',
        tol=1e-6, options=dict(disp=verbose, maxiter=max_iterations), callback=callback, **kwargs)

    return SimplexOptimizationResult(
        result=res.x, iterations=res.nit, simplexes=path,
        success=res.success,
        stop_reason=res.message
    )
