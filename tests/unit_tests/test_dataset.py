import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

    def test_dropna(self):
        X = np.array([
            [1.0, 2.0, np.nan],
            [4.0, 5.0, 6.0],
            [np.nan, 1.0, 2.0],
            [3.0, 4.0, 5.0]
        ])
        y = np.array([0, 1, 0, 1])

        ds = Dataset(X, y, features=['a','b','c'], label='y')
        ds_clean = ds.dropna()

        self.assertEqual(ds_clean.X.shape, (2, 3))
        self.assertEqual(ds_clean.y.shape, (2,))
        self.assertFalse(np.isnan(ds_clean.X).any())

    def test_fillna_with_mean(self):
        X = np.array([
            [1.0, 2.0, np.nan],
            [4.0, 5.0, 6.0]
        ])
        ds = Dataset(X, features=['a','b','c'])

        ds.fillna("mean")

        self.assertFalse(np.isnan(ds.X).any())
        self.assertAlmostEqual(ds.X[0,2], 6.0)

    def test_remove_by_index(self):
        X = np.array([
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ])
        y = np.array([10,20,30])
        ds = Dataset(X,y,features=['a','b','c'],label='y')

        ds.remove_by_index(1)

        self.assertEqual(ds.X.shape,(2,3))
        self.assertTrue((ds.X == np.array([[1,2,3],[7,8,9]])).all())
        self.assertTrue((ds.y == np.array([10,30])).all())

        # test negative index
        ds.remove_by_index(-1)
        self.assertEqual(ds.X.shape,(1,3))
        self.assertTrue((ds.X == np.array([[1,2,3]])).all())
