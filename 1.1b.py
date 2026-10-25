import numpy as np
matrix = np.array([[10, 20, 30],
                   [40, 50, 60],
                   [70, 80, 90]])
print(
    "Matrix:\n", matrix, "\n\n"
    "Element at [1,2]:", matrix[1, 2], "\n\n"
    "First Row:", matrix[0, :], "\n"
    "Second Column:", matrix[:, 1], "\n\n"
    "Top-left 2x2 Sub-matrix:\n", matrix[0:2, 0:2]
)