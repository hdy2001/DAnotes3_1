from problem1 import solve
import math
import numpy as np
import gmpy2


def func1(w):
    return solve(w) * (solve(w) + 1) * math.exp(solve(w))


def func2(w):
    return (2 * solve(w) + 1) * math.exp(solve(w))


def func3(w):
    return 2 * math.exp(solve(w))


def solve_1(mya, myb):
    return func1(myb) - func1(mya) - (func2(myb) -
                                      func2(mya)) + func3(myb) - func3(mya)


# def solve_2(mya, myb):
#     n = 100000
#     h = (myb - mya) / n
#     f_a = np.sqrt(solve(mya))
#     f_b = np.sqrt(solve(myb))
#     f_sum = 0
#     for i in range(n - 1):
#         print(i)
#         f_sum += np.sqrt(solve(mya + h * (i + 1)))
#     return (h / 2) * (f_a + 2 * f_sum + f_b)


def f(x):
    result = gmpy2.mpfr((2 * x * x + 2 * x * x * x * x) * np.exp(x * x))
    return result


def solve_2(mya, myb):
    n = 100000
    a = math.sqrt(solve(mya))
    b = math.sqrt(solve(myb))
    h = (b - a) / n
    f_a = f(a)
    f_b = f(b)
    f_sum = 0
    for i in range(n - 1):
        f_sum += f(a + h * (i + 1))
    return (h / 2) * (f_a + 2 * f_sum + f_b)


if __name__ == '__main__':
    a = solve_1(gmpy2.mpfr(0), gmpy2.mpfr(1))
    print(a)
    print(type(a))
    b = solve_2(gmpy2.mpfr(0), gmpy2.mpfr(1))
    print(b)
    print(type(b))
