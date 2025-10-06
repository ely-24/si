from si.base.transformer import Transformer
import numpy as np
from si.data.dataset import Dataset

class VarianceThreshold(Transformer):
    """
    Feature selector that removes all low-variance features.
    """

    def __init__(self, threshold: float = 0.4, **kwargs):
        """
        Initialize the VarianceThreshold feature selector.

        Parameters
        ----------
        threshold: float, default=0.0
            Features with a training-set variance lower than this threshold will be removed.
        """
        self.threshold = threshold
        self.variance = None

    def _fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Fit the VarianceThreshold feature selector to the data.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the feature selector to.

        Returns
        -------
        self: VarianceThreshold
            The fitted feature selector.
        """
        self.variance = np.var(dataset.X, axis =0)
        return self

    def _transform(self, dataset= Dataset) -> Dataset:
        
        mask = self.variance >= self.threshold
        X = dataset.X[:, mask]
        features = np.array(dataset.features)[mask]

        return Dataset(X=X, y=dataset.y, features=features, label=dataset.label)