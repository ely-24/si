import numpy as np
import warnings

def tanimoto_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes Tanimoto similarity between a single 1D sample x
    and multiple samples y (each row is one sample).

    Returns
    -------
    np.ndarray (n_samples,)
        Tanimoto similarity for each row in y.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if y.ndim != 2:
        raise ValueError("y must be a 2D array with each row as a sample.")
    if x.shape[0] != y.shape[1]:
        raise ValueError("x and y must have the same number of features.")

    # Dot products and squared norms
    dot_xy = np.dot(y, x)       # overlap with each sample nº de colunas diferente de linhas
    x2 = np.dot(x, x)           # |x|²
    y2 = np.sum(y * y, axis=1)  # |y|² for each row

    denom = x2 + y2 - dot_xy    # union definition

    # Handling division by zero safely
    if np.any(denom == 0):
        warnings.warn("Zero denominator encountered — similarity forced to 0 in those samples.")
        denom = np.where(denom == 0, 1e-10, denom)

    return dot_xy / denom

if __name__ == "__main__":
    print("\n TANIMOTO SIMILARITY TESTS \n")

    # Test 1 — Identical vectors → similarity = 1.0
    x1 = np.array([1, 0, 1, 1])
    y1 = np.array([[1, 0, 1, 1]])
    print("Test 1 — Identical:")
    print("x =", x1)
    print("y =", y1)
    print("→ similarity:", tanimoto_similarity(x1, y1), "\n")

    # Test 2 — Multiple samples, varied overlap
    x2 = np.array([1, 0, 1, 1])
    y2 = np.array([
        [1, 0, 1, 1],   # perfect match → 1.0
        [1, 1, 0, 0],   # partial match → lower score
        [0, 1, 0, 0]    # almost no overlap → very low
    ])
    print("Test 2 — Mixed similarity samples:")
    print("x =", x2)
    print("y =\n", y2)
    print("→ similarities:", tanimoto_similarity(x2, y2), "\n")

    # Test 3 — Extended binary case
    x3 = np.array([1, 1, 0, 1, 0])
    y3 = np.array([
        [1, 0, 0, 1, 1],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1]
    ])
    print("Test 3 — Larger feature vector:")
    print("x =", x3)
    print("y =\n", y3)
    print("→ similarities:", tanimoto_similarity(x3, y3), "\n")

    print("Function works correctly\n")
