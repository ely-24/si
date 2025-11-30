from typing import Callable, Union
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    K-Nearest Neighbors Regressor.
    
    Predicts a continuous value by averaging the values of the k-nearest 
    samples based on a distance metric.
    
    Parameters
    ----------
    k : int
        Number of neighbors to consider.
    distance : Callable
        Distance function between samples (default = euclidean_distance).

    Attributes (learned on fit)
    ----------
    dataset : Dataset
        Stores the training dataset used for neighbor search.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initializes the KNNRegressor.
        """
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> "KNNRegressor":
        """
        Training method for KNNRegressor.
        Training simply consists of storing the dataset.
        """
        # 1. Stores the training dataset
        self.dataset = dataset
        return self

    def _get_average_value(self, sample: np.ndarray) -> Union[int, float]:
        """
        Returns the average target value of the k-nearest neighbors of a sample.
        This is the central logic of k-NN regression.
        """
        # 1. Calculate the distance between the sample and all training samples
        distances = self.distance(sample, self.dataset.X)
        
        # 2. Get the indices of the k most similar examples (shortest distance)
        # argsort returns the indices that would sort the array. We take the first k.
        nearest_idxs = np.argsort(distances)[:self.k]
        
        # 3. Retrieve the corresponding target values (y) for these neighbors
        neighbor_values = self.dataset.y[nearest_idxs] 
        
        # 4. Calculate the average of the obtained values (Regression)
        return float(np.mean(neighbor_values)) 

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts continuous values for all samples in the dataset.
        """
        return np.apply_along_axis(self._get_average_value, 1, dataset.X)

    def _score(self, dataset: Dataset) -> float:
        """
        Calculates the error (RMSE) between the estimated values and the real values.
        """
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)