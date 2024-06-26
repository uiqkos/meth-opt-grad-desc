from collections.optional import Optional
from math import abs, sqrt


# struct Function[type: DType, N: Int, fn[(SIMD[DType.float64, N]), DType.float64]]:
#     var f: 



fn last[T: CollectionElement](arr: DynamicVector[T]) -> T:
    return arr[len(arr) - 1]

fn norm[N: Int](v: SIMD[DType.float64, N]) -> Float64:
    return sqrt((v ** 2).reduce_add())

fn gradient_descend[N: Int](
    func: fn(SIMD[DType.float64, N]) -> Float64,
    derivatives: StaticTuple[N, fn(SIMD[DType.float64, N]) -> Float64],
    start: SIMD[DType.float64, N],
    learning_rate: Float64,
    max_iter: Int,
    stop_function_delta: Optional[Float64] = None,
    stop_point_delta: Optional[Float64] = None,
) raises -> DynamicVector[SIMD[DType.float64, N]]:

    alias T = SIMD[DType.float64, N]
    
    if stop_function_delta and stop_function_delta.value() < 0:
        raise Error("Условие останова по значениям функции должно быть положительным")
    if stop_point_delta and stop_point_delta.value() < 0:
        raise Error("Условие останова по точкам должно быть положительным")
        
    var path = DynamicVector[T]()
    path.append(start)

    for _ in range(max_iter):
        var grad: SIMD[DType.float64, N] = SIMD[DType.float64, N]()

        for i in range(len(derivatives)):
            var coord = derivatives[i]
            grad[i] = coord(last(path))

        var new_point = last(path) - learning_rate * grad

        path.append(new_point)

        if stop_function_delta and abs(func(last(path)) - func(path[len(path) - 2])) < stop_function_delta.value():
            break

        if stop_point_delta and norm(last(path) - path[len(path) - 2]) < stop_point_delta.value():
            break

    return path



fn main() raises:
    fn f(x: SIMD[DType.float64, 2]) -> Float64:
        return (x ** 2).reduce_add()

    fn df_dx(x: SIMD[DType.float64, 2]) -> Float64:
        return 2 * x[0]

    fn df_dy(x: SIMD[DType.float64, 2]) -> Float64:
        return 2 * x[1]

    var d1: fn(SIMD[DType.float64, 2]) -> Float64 = df_dx
    var d2: fn(SIMD[DType.float64, 2]) -> Float64 = df_dy

    var path = gradient_descend(
        f,
        StaticTuple[2, _](d1, d2),
        start = SIMD[DType.float64, 2](1000, 1000),
        learning_rate = 0.0001,
        max_iter = 100000,
        stop_function_delta = Optional[Float64](1e-6),
    )
    
    print(len(path))

    # for i in range(len(path)):
        # print(path[i])