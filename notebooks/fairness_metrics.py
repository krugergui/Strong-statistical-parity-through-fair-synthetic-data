import numpy as np
from numba import njit


@njit
def DSP(predictions: np.array, groups: np.array) -> float:
    """
    Calculate the Disparate Impact of a predictive model.

    Args:
        predictions (List[any]): A list of predictions for each observation, 1 for positive and 0 for negative outcome.
        groups (List[any]): A list of group identifiers for each observation, 1 for previledged group and 0 for unpreviledged.

    Returns:
        float: The Disparate Impact score.
    """

    # assert predictions.shape[0] == groups.shape[0]
    # assert set(groups).issubset({0, 1})
    # assert set(predictions).issubset({0, 1})

    total_unpreviledged = np.sum(groups)
    total_previledged = groups.shape[0] - total_unpreviledged

    positive_unpreviledged = np.sum(groups & predictions)
    positive_previledged = np.sum(predictions) - positive_unpreviledged

    p_previledged = positive_previledged / total_previledged
    p_unpreviledged = positive_unpreviledged / total_unpreviledged

    # assert total_previledged > 0
    # assert total_unpreviledged > 0

    return abs(p_previledged - p_unpreviledged)


def DEO(model, X, Y, groups):
    # model: the trained model
    # X: our data of n examples with d features
    # Y: binary labels of our n examples (1 = positive)
    # groups: a list of n values binary values defining two different subgroups of the populations

    fY = model.predict(X)
    eo = [0, 0]
    eo[0] = float(
        len(
            [
                1
                for idx, fy in enumerate(fY)
                if fy == 1 and groups[idx] == 0 and Y[idx] == 1
            ]
        )
    ) / len([1 for idx, fy in enumerate(fY) if groups[idx] == 0 and Y[idx] == 1])
    eo[1] = float(
        len(
            [
                1
                for idx, fy in enumerate(fY)
                if fy == 1 and groups[idx] == 1 and Y[idx] == 1
            ]
        )
    ) / len([1 for idx, fy in enumerate(fY) if groups[idx] == 1 and Y[idx] == 1])
    return abs(eo[0] - eo[1])


@njit
def run_all_thresholds(predictions, list_protected):
    model_dsp = np.empty(
        101,
    )

    for i, threshold in enumerate(np.arange(0.0, 1.01, 0.01)):
        model_dsp[i] = DSP(np.where(predictions >= threshold, 1, 0), list_protected)

    return model_dsp
