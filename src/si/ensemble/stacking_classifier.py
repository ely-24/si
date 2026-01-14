import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier(Model):
    """
    Stacking ensemble classifier.
    Combines several base classifiers and trains a final classifier
    using the base models' predictions as features.
    """

    def __init__(self, base_models: list, meta_model: Model, **kwargs):
        super().__init__(**kwargs)
        self.base_models = base_models
        self.meta_model = meta_model

    def _fit(self, dataset: Dataset):
        """
        Fit base models and then fit the meta-model using their predictions.
        """

        # 1. Train base models
        for clf in self.base_models:
            clf.fit(dataset)

        # 2. Generate meta-features from base model predictions
        meta_features = []
        for clf in self.base_models:
            meta_features.append(clf.predict(dataset))

        meta_X = np.column_stack(meta_features)

        # 3. Create dataset for meta-model
        meta_dataset = Dataset(
            X=meta_X,
            y=dataset.y,
            features=[f"base_{i}" for i in range(len(self.base_models))],
            label=dataset.label
        )

        # 4. Train meta-model
        self.meta_model.fit(meta_dataset)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict labels using stacking ensemble.
        """

        meta_features = []
        for clf in self.base_models:
            meta_features.append(clf.predict(dataset))

        meta_X = np.column_stack(meta_features)

        meta_dataset = Dataset(
            X=meta_X,
            y=None,
            features=[f"base_{i}" for i in range(len(self.base_models))],
            label=None
        )

        return self.meta_model.predict(meta_dataset)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Compute accuracy of the stacking classifier.
        """
        return accuracy(dataset.y, predictions)
