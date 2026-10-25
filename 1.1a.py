import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(
    "Array a:", a, "\n"
    "Array b:", b, "\n\n"
    "Addition (a + b):", a + b, "\n"
    "Multiplication (a * b):", a * b, "\n"
    "Scalar Multiplication (a * 10):", a * 10, "\n\n"
    "Square Root of a:", np.sqrt(a), "\n"
    "Mean of a:", np.mean(a), "\n"
    "Dot Product (a · b):", np.dot(a, b)
)