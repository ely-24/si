from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Performs a stratified train-test split ensuring class proportions
    are maintained in both subsets.

    Parameters
    ----------
    dataset : Dataset
        Dataset object to be split
    test_size : float, optional
        Proportion of samples allocated to test data (default=0.2)
    random_state : int, optional
        Seed for reproducibility (default=42)

    Returns
    -------
    train_dataset : Dataset
        Stratified training dataset
    test_dataset : Dataset
        Stratified testing dataset
    """

    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")

    if not dataset.has_label():
        raise ValueError("Stratified split requires a labelled dataset.")

    np.random.seed(random_state)

    X, y = dataset.X, dataset.y
    unique_labels, counts = np.unique(y, return_counts=True)

    train_indices = []
    test_indices = []

    # Stratified sampling
    for label, count in zip(unique_labels, counts):
        label_indices = np.where(y == label)[0]
        np.random.shuffle(label_indices)

        n_test = round(count * test_size)  # keeps distribution accurate
        test_indices.extend(label_indices[:n_test])
        train_indices.extend(label_indices[n_test:])

    train_indices, test_indices = np.array(train_indices), np.array(test_indices)

    train_dataset = Dataset(X=X[train_indices], y=y[train_indices], 
                            features=dataset.features, label=dataset.label)

    test_dataset = Dataset(X=X[test_indices], y=y[test_indices], 
                            features=dataset.features, label=dataset.label)

    return train_dataset, test_dataset


