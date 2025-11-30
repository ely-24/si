import unittest
import numpy as np
from si.data.dataset import Dataset
from si.feature_selection.select_percentile import SelectPercentile


class TestSelectPercentile(unittest.TestCase):

    def setUp(self):
        # synthetic dataset for controlled testing
        self.X = np.array([
            [1,2,3,4],
            [5,6,7,8],
            [1,3,5,7],
            [2,4,6,8]
        ])
        self.y = np.array([0,1,0,1])
        self.features = ["f1","f2","f3","f4"]

    def test_select_percentile_50(self):
        """Should retain ~50% of features (2 out of 4)"""
        ds = Dataset(self.X, self.y, self.features, "y")

        selector = SelectPercentile(percentile=50)
        ds_transformed = selector.fit_transform(ds)

        self.assertEqual(ds_transformed.X.shape[1], 2)
        self.assertEqual(len(ds_transformed.features), 2)

    def test_select_percentile_25_selects_one_feature(self):
        """25% of 4 features = 1 feature retained"""
        ds = Dataset(self.X, self.y, self.features, "y")

        selector = SelectPercentile(percentile=25)
        ds_transformed = selector.fit_transform(ds)

        self.assertEqual(ds_transformed.X.shape[1], 1)

    def test_select_percentile_100_returns_all(self):
        """100% selection means no feature removal"""
        ds = Dataset(self.X, self.y, self.features, "y")

        selector = SelectPercentile(percentile=100)
        ds_transformed = selector.fit_transform(ds)

        self.assertEqual(ds_transformed.X.shape[1], 4)
        self.assertListEqual(ds_transformed.features, self.features)
