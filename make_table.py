import tqdm

import utils as u
from all import *
import pandas as pd
from typing_local import *

x, y = sympy.symbols('x y')
funcs = [
    x ** 2 + y ** 2,
    2 * (y - x ** 2) ** 2 + (1 - x) ** 2,
    sp.sin(x) * sp.sin(y) + 0.01 * x ** 2 + 0.01 * y ** 2,
    (sp.sin(x) * sp.sin(y) + 0.01 * x ** 2 + 0.01 * y ** 2) * x ** 2,
    -20 * sp.exp(-0.2 * sp.sqrt(0.5 * (x ** 2 + y ** 2))) - sp.exp(
        0.5 * (sp.cos(2 * sp.pi * x) + sp.cos(2 * sp.pi * y))) + 20 + sp.exp(1),
    sp.sin(3 * sp.pi * x) ** 2 + (x - 1) ** 2 * (1 + sp.sin(3 * sp.pi * y) ** 2) + (y - 1) ** 2 * (
            1 + sp.sin(2 * sp.pi * y) ** 2),
    -0.0001 * (sp.sin(x) * sp.sin(y) * sp.exp(100 - sp.sqrt(x ** 2 + y ** 2) / sp.pi)) ** 0.1 + 1,
    0.5 + (sp.sin(x ** 2 - y ** 2) ** 2 - 0.5) / ((1 + 0.001 * (x ** 2 + y ** 2)) ** 2),
    -(y + 47) * sp.sin(sp.sqrt(sp.sqrt((x / 2 + (y + 47)) ** 2))) * sp.sin(sp.sqrt(sp.sqrt((x - (y + 47)) ** 2))),
    -sp.cos(x) * sp.cos(y) * sp.exp(-((x - sp.pi) ** 2 + (y - sp.pi) ** 2))

    # y + sympy.log(x ** 2 + y ** 2),
]


def from_sympy_to_plotable_func(f):
    return fs.tupled(np.vectorize(sympy.lambdify([x, y], f, 'numpy')))


data = []

for f in tqdm.tqdm(funcs):
    for start in [(1, 1), (10, 10), (100, 100)]:
        for method in methods:
            for lr in [constant_rate(0.001), dichotomy_method(0.001), golden_ratio_method(0.001)]:
                record = method(f, start, lr)
                # match record:
                #     case NewtonOptimizationResult(result, iters, stop, success, path) if iters < 10 and f == funcs[
                #         0] and False:
                #         u.plot_func(from_sympy_to_plotable_func(f), path, label=str(f) + ' ' + method.__name__).show()
                #     case GradientOptimizationResult(result, iters, stop, success, path) if f == funcs[1]:
                #         u.plot_func(from_sympy_to_plotable_func(f), path, label=str(f) + ' ' + method.__name__).show()
                #
                data.append({
                    'function': str(f),
                    'start_point': str(start),
                    'method': method.__name__,
                    'success': record.success,
                    'iterations': record.iterations,
                    'result': record.result,
                    'stop_reason': record.stop_reason,
                    'path_len': len(record.path if hasattr(record, 'path') else record.simplexes),
                    'path': record.path if hasattr(record, 'path') else record.simplexes,
                    'lr': lr.__name__
                })

pd.DataFrame(data).to_csv('results_total.csv')
