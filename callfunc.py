from numba import njit


@njit
def g(a, b):
    return a + b


@njit
def f(a, b):
    return g(a, b)


print(f(1, 2))
