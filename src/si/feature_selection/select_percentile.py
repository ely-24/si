import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification 


class SelectPercentile(Transformer):

    def __init__(self, percentile: float, score_func: callable = f_classification, **kwargs):
        """
        Selects the top percentage of features based on F-score ranking.

        Parameters
        ----------
        percentile : float
            Percent of features to keep (0 < p ≤ 100).
        score_func : callable
            Score function returning (F, p) for each feature.
        """
        super().__init__(**kwargs)

        if not 0 < percentile <= 100:
            raise ValueError("Percentile must be in the interval [0,100].")

        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> "SelectPercentile":
        """Estimate statistical scores for each feature."""
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects features whose F-score ranks within the desired percentile.
        Handles score ties to guarantee the correct number of selected features.
        """

        n_features = len(self.F)
        target_n = max(1, int(round((self.percentile/100) * n_features)))

        # compute percentile cutoff
        cutoff = np.percentile(self.F, 100 - self.percentile)

        # initial selection using threshold
        initial_mask = self.F >= cutoff
        selected = np.where(initial_mask)[0]

        # if too many features were selected (tie at cutoff), trim by ranking
        if len(selected) > target_n:
            order = np.argsort(self.F)[::-1]           
            selected = order[:target_n]

        selected = np.sort(selected)

        return Dataset(
            X=dataset.X[:, selected],
            y=dataset.y,
            features=[dataset.features[i] for i in selected],
            label=dataset.label
        )
