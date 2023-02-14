from numba import njit


@njit
def f(a, b):
    return a + b


print(f(1, 2))
