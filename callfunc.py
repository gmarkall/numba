from numba import njit


@njit(_nrt=False)
def g(a, b):
    return a + b


@njit(_nrt=False)
def f(a, b):
    return g(a, b)


print(f(1, 2))
