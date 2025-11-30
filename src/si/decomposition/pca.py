import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class PCA(Transformer):
    """
    Principal Component Analysis (PCA) using eigenvalue decomposition.
    """

    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.mean = None
        self.components = None           
        self.explained_variance = None   


    def _fit(self, dataset: Dataset):
        """
        Computes the principal components using eigenvalue decomposition.
        """

        X = dataset.X

        # 1. Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Covariance matrix + eigen decomposition
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # 3. Sort eigenvectors by descending variance
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # 4. Store principal components AS ROWS (as required by the exercise)
        # Transpose so that each principal component is a row (shape: n_components, n_features)
        self.components = eigenvectors[:, :self.n_components].T

        # 5. Explained variance ratio (first n eigenvalues / total)
        total_variance = np.sum(eigenvalues)
        self.explained_variance = eigenvalues[:self.n_components] / total_variance

        return self

    def _transform(self, dataset: Dataset):
        """
        Projects the dataset onto the principal components space.
        """

        if self.mean is None or self.components is None:
            raise ValueError("PCA instance is not fitted yet. Call 'fit' before 'transform'.")

        X_centered = dataset.X - self.mean
        X_reduced = np.dot(X_centered, self.components.T)

        return Dataset(
            X=X_reduced,
            y=dataset.y,
            features=[f"PC{i+1}" for i in range(self.n_components)],
            label=dataset.label
        )
