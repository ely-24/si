from typing import Callable, Union
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    K-Nearest Neighbors Regressor
    
    Predicts a continuous value by averaging the values of the k-nearest
    samples based on a distance metric.
    
    Parameters
    ----------
    k : int
        Number of neighbors to use.
    distance : Callable
        Distance function between samples (default = euclidean_distance).

    Attributes (learned on fit)
    ----------
    dataset : Dataset
        Stored training dataset used for neighbor search.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> "KNNRegressor":
        """
        Stores the training dataset.
        """
        self.dataset = dataset
        return self

    def _get_average_value(self, sample: np.ndarray) -> Union[int, float]:
        """
        Returns the average target value of the k-nearest neighbors of a sample.
        """
        distances = self.distance(sample, self.dataset.X)                   
        nearest_idxs = np.argsort(distances)[:self.k]                       
        neighbor_values = self.dataset.y[nearest_idxs]                      

        return float(np.mean(neighbor_values))                              

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts continuous values for all samples in dataset.
        """
        return np.apply_along_axis(self._get_average_value, 1, dataset.X)

    def _score(self, dataset: Dataset) -> float:
        """
        Returns RMSE between predictions and true values.
        """
        preds = self.predict(dataset)
        return rmse(dataset.y, preds)
