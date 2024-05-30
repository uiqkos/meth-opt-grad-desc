import scipy

from typing_local import *


def add_sim():
    import inspect

    try:
        nelder_code = inspect.getsource(scipy.optimize._optimize._minimize_neldermead)
        minimize_code = inspect.getsource(scipy.optimize.minimize)
    except OSError:
        return

    marker = 'intermediate_result = OptimizeResult(x=sim[0], fun=fsim[0])'
    sim_code = "intermediate_result['sim'] = sim"

    if sim_code in nelder_code:
        return

    idx = nelder_code.index(marker) + len(marker)
    nelder_code = nelder_code[:idx] + '; ' + sim_code + nelder_code[idx:]

    ns = {}
    exec(nelder_code, scipy.optimize._optimize.__dict__, ns)  # exec(code, globals, locals)

    exec(minimize_code,
         scipy.optimize._minimize.__dict__ | ns, ns)  # ns = {'_minimize_neldermead': ...}

    scipy.optimize.minimize = ns['minimize']  # ns = {'_minimize_neldermead': ..., 'minimize': ...}


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
