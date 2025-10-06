import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset 
from typing import Callable


class SelectKBest (Transformer):
    def __init__(self, k: int, statistical_test: callable, **kwargs):
        self.k = k
        self.F = None
        self.p = None
        self.statistical_test = statistical_test
    
    def _fit(self, dataset: Dataset) -> 'SelectKBest':
        self.F, self.p = self.score_func(dataset)
    
    def _transform(self, dataset: Dataset) -> Dataset:
        idx = np.argsort(self.F)[-self.k:]
        X = dataset.X[:, idx]
        features = dataset.features[idx]

        return Dataset(X=X, y=dataset.y, features=features, label=dataset.label)