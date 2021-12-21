from problem1 import solve
import math
"""
解析表达式的意思是可以选择多种积分公式吗还是可以写出明确的表达式？
"""


def func1(w):
    return solve(w) * (solve(w) + 1) * math.exp(solve(w))


def func2(w):
    return (2 * solve(w) + 1) * math.exp(solve(w))


def func3(w):
    return 2 * math.exp(solve(w))


def solve_1(mya, myb):
    return func1(myb) - func1(mya) - (func2(myb) -
                                      func2(mya)) + func3(myb) - func3(mya)


def solve_2(mya, myb):
    n = 200000
    h = myb / n
    f_a = solve(mya)
    f_b = solve(myb)
    f_sum = 0
    for i in range(n - 1):
        f_sum += solve(mya + h * (i + 1))
    return (h / 2) * (f_a + 2 * f_sum + f_b)


if __name__ == '__main__':
    print(solve_1(0, 1))
    print(solve_2(0, 1))
