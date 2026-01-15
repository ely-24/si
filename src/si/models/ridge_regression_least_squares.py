import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression estimator solved with the closed-form analytical solution.

    This method introduces an L2 penalty to shrink coefficients and reduce variance.
    The parameters are obtained via:

        θ = (XᵀX + λI)⁻¹ Xᵀy

    where λ controls the strength of regularization.
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        """
        Parameters
        ----------
        l2_penalty : float
            Strength of the L2 regularization term.
        scale : bool
            Standardize features before fitting.
        """
        super().__init__(**kwargs)
        self.lambda_ = l2_penalty
        self.scale = scale

        # learned parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None


    def _fit(self, dataset: Dataset) -> "RidgeRegressionLeastSquares":
        """
        Estimate model parameters using ridge closed-form solution.
        """

        X = dataset.X.copy()
        y = dataset.y

        # 1 ─ Optional standardization
        if self.scale:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
            X = (X - self.mean) / self.std

        n_samples, n_features = X.shape

        # 2 ─ Add intercept column
        X_aug = np.c_[np.ones(n_samples), X]       # shape → (n, p+1)

        # 3 ─ Build regularization matrix
        reg = self.lambda_ * np.eye(n_features + 1)
        reg[0, 0] = 0                               # do not penalize intercept

        # 4 ─ Solve closed-form ridge solution
        A = X_aug.T @ X_aug + reg #mais eficiente que usar o np.dot
        B = X_aug.T @ y
        coefs = np.linalg.inv(A) @ B

        # store parameters
        self.theta_zero = coefs[0]
        self.theta = coefs[1:]

        return self


    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict y using learned parameters.
        """

        X = dataset.X.copy()

        # standardize using training parameters
        if self.scale:
            X = (X - self.mean) / self.std

        X_aug = np.c_[np.ones(X.shape[0]), X]
        weights = np.r_[self.theta_zero, self.theta]

        return X_aug @ weights


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Compute model MSE error.
        """
        return mse(dataset.y, predictions)



# Quick minimal usage example
if __name__ == "__main__":
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    ds = Dataset.from_random(200, 5)
    train, test = train_test_split(ds, 0.2)

    model = RidgeRegressionLeastSquares(l2_penalty=1.0)
    model.fit(train)

    preds = model.predict(test)
    print("Predictions sample:", preds[:5])
    print("Real values sample:", test.y[:5])
    print("MSE:", model.score(test))
