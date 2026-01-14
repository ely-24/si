import numpy as np
import itertools
from si.base.model import Model
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation


def randomized_search_cv(
    model: Model,
    dataset: Dataset,
    hyperparameter_grid: dict,
    cv: int = 3,
    n_iter: int = 10,
    scoring: callable = None
) -> dict:
    """
    Randomized Search with Cross-Validation for hyperparameter tuning.
    """

    # 1. Validate hyperparameters
    for param_name in hyperparameter_grid.keys():
        if not hasattr(model, param_name):
            raise AttributeError(
                f"Model {type(model).__name__} has no hyperparameter '{param_name}'"
            )

    # 2. Generate all possible combinations
    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())
    all_configs = list(itertools.product(*param_values))

    # 3. Randomly select configurations
    n_iter = min(n_iter, len(all_configs))
    chosen_indices = np.random.choice(len(all_configs), size=n_iter, replace=False)

    scores = []
    tested_configs = []

    # 4. Evaluate each configuration
    for idx in chosen_indices:
        config = dict(zip(param_names, all_configs[idx]))

        for param, value in config.items():
            setattr(model, param, value)

        cv_scores = k_fold_cross_validation(
            model=model,
            dataset=dataset,
            scoring=scoring,
            cv=cv
        )

        scores.append(np.mean(cv_scores))
        tested_configs.append(config)

    # 5. Select best configuration
    best_position = int(np.argmax(scores))

    return {
        "hyperparameters": tested_configs,
        "scores": scores,
        "best_hyperparameters": tested_configs[best_position],
        "best_score": scores[best_position]
    }
