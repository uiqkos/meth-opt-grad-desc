import time

import numpy as np

from scipy.optimize import minimize


def rosenbrock(point):
    return (1 - point[0]) ** 2 + 100 * (point[1] - point[0] ** 2) ** 2


def quadratic(point):
    return (point[0] - 2) ** 2 + (point[1] + 3) ** 2


def heavy_quadratic(point):
    return (point[0] - 2) ** 2 + 0.0004 * (point[1] + 3) ** 2


def rosenbrock_grad(point):
    dfdx0 = -2 * (1 - point[0]) - 400 * point[0] * (point[1] - point[0] ** 2)
    dfdx1 = 2 * 100 * (point[1] - point[0] ** 2)
    return np.array([dfdx0, dfdx1])


def quadratic_grad(point):
    dfdx0 = 2 * (point[0] - 2)
    dfdx1 = 2 * (point[1] + 3)
    return np.array([dfdx0, dfdx1])


def heavy_quadratic_grad(point):
    x, y = point
    df_dx = 2 * (x - 2)
    df_dy = 0.0008 * (y + 3)
    return np.array([df_dx, df_dy])


def gradient_descent(grad_function, start_point, learning_rate=0.001, tolerance=1e-6, max_iterations=1000000,
                     max_evals=10000000):
    point = np.array(start_point, dtype=np.float16)
    evals = 0
    it = 0
    for _ in range(max_iterations):
        grad = grad_function(point)
        point_new = point - learning_rate * grad

        if np.linalg.norm(point_new - point) < tolerance:
            break
        '''if grad_function == rosenbrock_grad:
            if np.linalg.norm(point_new - np.array([1, 1])) < tolerance:
                break
        if grad_function == quadratic or grad_function == heavy_quadratic_grad:
            if np.linalg.norm(point_new - np.array([2, -3])) < tolerance:
                break'''
        point = point_new
        evals += 1
        it += 1
        if evals >= max_evals:
            break
    #print("gd-iterations:" + str(it))
    #print("gd-evaluations:" + str(evals))
    return point, it


def gradient_descent_dichotomy(function, grad_function, start_point, tolerance=1e-6, max_iterations=10000000,
                               max_evals=3000000):
    point = np.array(start_point, dtype=np.float16)
    evals = 0
    it = 0
    for _ in range(max_iterations):
        it += 1
        grad = grad_function(point)

        def f_new(alpha):
            return function(point - alpha * grad)

        alpha_left, alpha_right = 0, 1
        while alpha_right - alpha_left > tolerance:
            alpha_mid = (alpha_left + alpha_right) / 2
            if f_new(alpha_mid - tolerance / 2) < f_new(alpha_mid + tolerance / 2):
                alpha_right = alpha_mid
            else:
                alpha_left = alpha_mid
            evals += 3
            if evals >= max_evals:
                break
        alpha_optimal = (alpha_left + alpha_right) / 2
        point_new = point - alpha_optimal * grad
        '''if function == rosenbrock:
            if np.linalg.norm(point_new - np.array([1, 1])) < tolerance:
                break
        if function == quadratic or function == heavy_quadratic:
            if np.linalg.norm(point_new - np.array([2, -3])) < tolerance:
                break'''

        if np.linalg.norm(point_new - point) < tolerance:
            break
        point = point_new

    #print("dich-iterations:" + str(it))
    #print("dich-evaluations:" + str(evals))
    return point, it


def nelder_mead(function, start_point, precision):
    options = {
        'xatol': precision,
        'maxiter': 1000000,
        'maxfev': 300000
    }
    result = minimize(function, start_point, method='Nelder-Mead', options=options)
    return result.x


def gradient_descent_golden_section(function, grad_function, start_point, tolerance=1e-6,
                                    max_iterations=100000):
    golden_ratio = (np.sqrt(5) - 1) / 2
    point = np.array(start_point, dtype=np.float16)
    evals = 0
    it = 0

    for _ in range(max_iterations):
        grad = grad_function(point)
        it += 1
        evals += 2

        a, b = 0, 1
        c = b - golden_ratio * (b - a)
        d = a + golden_ratio * (b - a)
        while abs(c - d) > tolerance:
            if function(point - c * grad) < function(point - c * grad):
                b = d
            else:
                a = c
            c = b - golden_ratio * (b - a)
            d = a + golden_ratio * (b - a)
        alpha_optimal = (b + a) / 2

        point_new = point - alpha_optimal * grad
        if np.linalg.norm(point_new - point) < tolerance:
            break
        '''if function == rosenbrock:
            if np.linalg.norm(point_new - np.array([1, 1])) < tolerance:
                break
        if function == quadratic or function == heavy_quadratic:
            if np.linalg.norm(point_new - np.array([2, -3])) < tolerance:
                break'''
        point = point_new
    #print("golden-r-iterations:" + str(it))
    #print("golden-r-evaluations:" + str(evals))

    return point, it


#start_point_rosenbrock = np.array([0, 0])
#start_point_quadratic = np.array([0, 0])
start_point_rosenbrock = np.array([-1.0, 2.0])
start_point_quadratic = np.array([0.5, -2.0])

results = {}
'''print("----------POINTS TOLERANCE/PRECISION----------")
for required_tolerance in [1e-3, 1e-4, 1e-5, 1e-6]:
    print("=========================================================")
    print("required tolerance:" + str(required_tolerance))
    #print("required precision:" + str(required_tolerance))
    print()
    print("for rosenbrock:")
    gd_rosenbrock = gradient_descent(rosenbrock_grad, start_point_rosenbrock, 0.001, required_tolerance)
    gd_rosenbrock_dichotomy = gradient_descent_dichotomy(rosenbrock, rosenbrock_grad, start_point_rosenbrock,
                                                         required_tolerance)
    nm_rosenbrock = nelder_mead(rosenbrock, start_point_rosenbrock, required_tolerance)
    gr_rosenbrock = gradient_descent_golden_section(rosenbrock, rosenbrock_grad, start_point_rosenbrock,
                                                    required_tolerance)
    print()
    print("for quadratic:")
    gd_quadratic = gradient_descent(quadratic_grad, start_point_quadratic, 0.001, required_tolerance)
    gd_quadratic_dichotomy = gradient_descent_dichotomy(quadratic, quadratic_grad, start_point_quadratic,
                                                        required_tolerance)
    nm_quadratic = nelder_mead(quadratic, start_point_quadratic, required_tolerance)
    gr_quadratic = gradient_descent_golden_section(quadratic, quadratic_grad, start_point_quadratic,
                                                   required_tolerance)
    print()
    print("for heavy quadratic:")
    gd_hquadratic = gradient_descent(heavy_quadratic_grad, start_point_quadratic, 0.001, required_tolerance)
    gd_hquadratic_dichotomy = gradient_descent_dichotomy(heavy_quadratic, heavy_quadratic_grad, start_point_quadratic,
                                                         required_tolerance)
    nm_hquadratic = nelder_mead(heavy_quadratic, start_point_quadratic, required_tolerance)
    gr_hquadratic = gradient_descent_golden_section(heavy_quadratic, heavy_quadratic_grad, start_point_quadratic,
                                                    required_tolerance)
    print()
    results["rosenbrock"] = {
        "gd": gd_rosenbrock,
        "gd_dichotomy": gd_rosenbrock_dichotomy,
        "nm": nm_rosenbrock,
        "golden_ration": gr_rosenbrock
    }

    results["quadratic"] = {
        "gd": gd_quadratic,
        "gd_dichotomy": gd_quadratic_dichotomy,
        "nm": nm_quadratic,
        "golden_ration": gr_quadratic
    }

    results["heavy-quadratic"] = {
        "gd": gd_hquadratic,
        "gd_dichotomy": gd_hquadratic_dichotomy,
        "nm": nm_hquadratic,
        "golden_ration": gr_hquadratic
    }
    for key in results:
        print(str(key) + "values:")
        for method in results[key]:
            print(method, *results[key][method])
        print()'''

print("----------STARTING POINTS COMPARING----------")
for d in [0]:
    print("=========================================================")
    print("start rosenbrock:", str(start_point_rosenbrock[0] + d) + "," + str(start_point_rosenbrock[1] + d))
    print()
    print("start quadratic:", str(start_point_quadratic[0] + d) + "," + str(start_point_quadratic[1] + d))
    print()


    st = time.perf_counter()
    gd_quadratic, it = gradient_descent(quadratic_grad, start_point_quadratic + d)
    print(it)
    print("gd-quadratic:", (time.perf_counter() - st)*1000)
    st = time.perf_counter()
    gd_quadratic_dichotomy, it = gradient_descent_dichotomy(quadratic, quadratic_grad, start_point_quadratic + d)
    print(it)
    print("dich-quadratic:", (time.perf_counter() - st)*1000)
    st = time.perf_counter()
    gr_quadratic, it = gradient_descent_golden_section(quadratic, quadratic_grad, start_point_quadratic + d)
    print(it)
    print("golden-quadratic:", (time.perf_counter() - st)*1000)
    st = time.perf_counter()
    gd_rosenbrock, it = gradient_descent(rosenbrock_grad, start_point_rosenbrock + d)
    print(it)
    print("gd-rosenbrock:", (time.perf_counter() - st)*1000)
    st = time.perf_counter()
    gd_rosenbrock_dichotomy, it = gradient_descent_dichotomy(rosenbrock, rosenbrock_grad, start_point_rosenbrock + d)
    print(it)
    print("dich-rosenbrock:", (time.perf_counter() - st)*1000)
    st = time.perf_counter()
    gr_rosenbrock, it = gradient_descent_golden_section(rosenbrock, rosenbrock_grad, start_point_rosenbrock + d)
    print(it)
    print("golden-rosenbrock:", (time.perf_counter() - st)*1000)
    st = time.perf_counter()
    gd_hquadratic, it = gradient_descent(heavy_quadratic_grad, start_point_quadratic + d)
    print(it)
    print("gd-heavy-quadratic:", (time.perf_counter() - st)*1000)
    st = time.perf_counter()
    gd_hquadratic_dichotomy, it = gradient_descent_dichotomy(heavy_quadratic, heavy_quadratic_grad, start_point_quadratic + d)
    print(it)
    print("dich-heavy-quadratic:", (time.perf_counter() - st)*1000)
    st = time.perf_counter()
    gr_hquadratic, it = gradient_descent_golden_section(heavy_quadratic, heavy_quadratic_grad, start_point_quadratic + d)
    print(it)
    print("golden-heavy-quadratic:", (time.perf_counter() - st)*1000)
    results["rosenbrock"] = {
        "gd": gd_rosenbrock,
        "gd_dichotomy": gd_rosenbrock_dichotomy,
        "golden_ration": gr_rosenbrock
    }

    results["quadratic"] = {
        "gd": gd_quadratic,
        "gd_dichotomy": gd_quadratic_dichotomy,
        "golden_ration": gr_quadratic
    }

    results["heavy-quadratic"] = {
        "gd": gd_hquadratic,
        "gd_dichotomy": gd_hquadratic_dichotomy,
        "golden_ration": gr_hquadratic
    }
    for key in results:
        print(str(key) + " values:")
        for method in results[key]:
            print(method, *results[key][method])
        print()