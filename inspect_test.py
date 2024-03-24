import inspect
import inspect_tes2


if __name__ == '__main__':
    exec(compile(inspect.getsource(inspect_tes2.f2).replace('1', '2'), '<object>', 'exec'))
    inspect_tes2.f2 = f2
    print(f2(3))
    print(inspect_tes2.f1(4))
