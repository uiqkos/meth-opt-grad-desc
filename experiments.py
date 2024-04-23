import inspect

import numpy as np
import pandas as pd
import scipy.optimize
import sympy

from main import gradient_descend, dichotomy_method, constant_rate, scipy_nelder_mead, golden_ratio_method
from utils import R2_derivatives, gen_func, plot_func, call_counter, plot_func_with_path_matplotlib, plot_2d_with_color, \
    plot_2d_and_3d_side_by_side


def experiment(func_coefs, funcs, points, methods):
    records = []

    for coefs, func in zip(func_coefs, funcs):
        dfs = R2_derivatives(coefs)
        for start in map(np.array, points):
            for method in methods:
                dfs_with_counter = [call_counter(d) for d in dfs]
                func_with_counter = call_counter(func)
                res = method(start, dfs_with_counter, func_with_counter)
                records.append({
                    'coefs': coefs,
                    'start': start,
                    'method': method.__name__,
                    'target func calls': func_with_counter.calls,
                    'gradient func calls': tuple(d.calls for d in dfs_with_counter),
                    'minimum': res.minimum,
                    'path': res.path,
                    'iterations': res.iterations,
                })
                title = fr'{method.__name__} ${sympy.latex(func(sympy.symbols("x y")))}$ start = {start}'
                text = title + f'\niterations: {res.iterations}\n' \
                               f'target func calls: {func_with_counter.calls}\n' \
                               f'gradient func calls: {tuple(d.calls for d in dfs_with_counter)}'
                plot_2d_and_3d_side_by_side(
                    func,
                    res.path if res.path is not None else res.simplexes,
                    limit=max(start) * 1.5, issimplex=method == scipy_nelder_mead_,
                    title=title, save=False, text=text
                )
                # plot_func(func, res.path).show()
