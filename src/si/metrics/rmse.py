import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error (RMSE) metric.

    Measures the average magnitude of prediction errors.
    Lower values indicate better performance.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    float
        RMSE score
    """

    return np.sqrt(np.mean((y_true - y_pred) ** 2))

