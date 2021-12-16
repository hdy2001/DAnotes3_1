import math


# W'(z)
def f(x, y):
    return 1 / (x + math.exp(y))


def solve(myZ):
    """
    myZ: 大于0的任意float类型
    """
    # W(0) = 0
    x_n = 0
    y_n = 0
    h = 0.001
    # 迭代公式
    for _ in range(int(myZ / h)):
        y_bar_n_plus_1 = y_n + h * f(x_n, y_n)
        x_n_plus_1 = x_n + h
        y_n_plus_1 = y_n + h / 2 * (f(x_n, y_n) +
                                    f(x_n_plus_1, y_bar_n_plus_1))

        # 更新参数
        x_n = x_n_plus_1
        y_n = y_n_plus_1

    return y_n


if __name__ == '__main__':
    print(solve(2))
