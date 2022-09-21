import numpy as np


def exact(y, arg_grid):
    values = np.zeros(len(arg_grid))
    for i in range(0, len(arg_grid)):
        values[i] = y(arg_grid[i])
    return values


def euler(f, arg_grid, h, init_value):
    values = np.zeros(len(arg_grid))
    values[0] = init_value
    for i in range(0, len(arg_grid) - 1):
        values[i + 1] = values[i] + h * f(arg_grid[i], values[i])
    return values


def euler_cauchy(f, arg_grid, h, init_value):
    values = np.zeros(len(arg_grid))
    values[0] = init_value
    for i in range(1, len(arg_grid)):
        f_prev = f(arg_grid[i - 1], values[i - 1])
        val = values[i - 1] + h * f_prev
        values[i] = values[i - 1] + h * (f_prev + f(arg_grid[i - 1], val)) / 2
    return values


def runge_kutta4(f, arg_grid, h, init_value):
    values = np.zeros(len(arg_grid))
    values[0] = init_value

    for i in range(1, len(arg_grid)):

        x_prev = arg_grid[i - 1]
        y_prev = values[i - 1]
        k1 = h * f(x_prev, y_prev)
        k2 = h * f(x_prev + h / 2, y_prev + k1 / 2)
        k3 = h * f(x_prev + h / 2, y_prev + k2 / 2)
        k4 = h * f(x_prev + h, y_prev + k3)
        dy = (k1 + 2*k2 + 2*k3 + k4) / 6
        values[i] = y_prev + dy
    return values


def adams(f, arg_grid, h, init_values):
    values = np.zeros(len(arg_grid))
    for i in range(len(init_values)):
        values[i] = init_values[i]

    for i in range(len(init_values), len(arg_grid)):
        values[i] = values[i - 1] + h/24 * \
                    (
                    55*f(arg_grid[i - 1], values[i - 1]) -
                    59*f(arg_grid[i - 2], values[i - 2]) +
                    37*f(arg_grid[i - 3], values[i - 3]) -
                    9*f(arg_grid[i - 4], values[i - 4])
                    )
    return values