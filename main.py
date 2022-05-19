import numpy as np


def multiply(a, b):
    result = np.einsum("ik,kj->ij", a, b)
    print("multiply", result, sep="\n")
    print()
    return result


def transpose(matrix):
    return np.einsum("ij->ji", matrix)


def trace(matrix):
    return np.einsum("ii->", matrix)


if __name__ == '__main__':
    # random numbers
    # A = np.random.sample((3, 2))
    # B = np.random.sample((2, 2))

    # not random numbers:)
    A = [[1, 1], [2, 2], [3, 3]]
    B = [[1, 1], [2, 2]]
    print(A, B)
    ABBAt = multiply(multiply(multiply(A, B), B), transpose(A))
    trace = trace(ABBAt)
    print(trace)
