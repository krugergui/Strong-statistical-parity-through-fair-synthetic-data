import numpy as np
from numba import njit


@njit
def DSP(predictions: np.array, groups: np.array) -> float:
    """
    Calculate the Disparate Impact of a predictive model.
    P ( Y = 1 | X, S=s1) - P ( Y = 1 | X, S=s2)

    Args:
        predictions (np.array): The predictions of the model. 1 for positive and 0 for negative.
        groups (np.array): The groups of the individuals. 1 for previledged and 0 for unpreviledged.

    Returns:
        float: The Disparate Impact score.
    """

    assert predictions.shape[0] == groups.shape[0]

    total_previledged = 0
    total_unpreviledged = 0
    positive_unpreviledged = 0
    positive_previledged = 0

    for grp, pred in zip(groups, predictions):
        if pred != 0:
            if pred != 1:
                raise ValueError("Prediction must be 0 or 1.")

        if grp == 0:
            total_unpreviledged += 1
            if pred == 1:
                positive_unpreviledged += 1
        elif grp == 1:
            total_previledged += 1
            if pred == 1:
                positive_previledged += 1
        else:
            raise ValueError("Group identifier must be 0 or 1.")

    p_previledged = positive_previledged / total_previledged
    p_unpreviledged = positive_unpreviledged / total_unpreviledged

    assert total_previledged > 0
    assert total_unpreviledged > 0

    return abs(p_previledged - p_unpreviledged)


@njit
def run_all_thresholds(predictions, list_protected):
    model_dsp = np.empty(
        101,
    )

    for i, threshold in enumerate(np.arange(0.0, 1.01, 0.01)):
        model_dsp[i] = DSP(np.where(predictions >= threshold, 1, 0), list_protected)

    return model_dsp
