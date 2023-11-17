import time

import numpy as np
from numba import njit, prange

PREVILEDGED = 1
UNPREVILEDGED = 0
POSITIVE_OUTCOME = 1
NEGATIVE_OUTCOME = 0


@njit("float64(int64[:], int64[:])", parallel=True)
def calculcate_demografic_statistical_parity(predictions, groups) -> float:
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

    for idx in prange(len(predictions)):
        if groups[idx] == UNPREVILEDGED:
            total_unpreviledged += 1
            if predictions[idx] == POSITIVE_OUTCOME:
                positive_unpreviledged += 1
        elif groups[idx] == PREVILEDGED:
            total_previledged += 1
            if predictions[idx] == POSITIVE_OUTCOME:
                positive_previledged += 1

    p_previledged = positive_previledged / total_previledged
    p_unpreviledged = positive_unpreviledged / total_unpreviledged

    return np.abs(p_previledged - p_unpreviledged)


@njit("UniTuple(float64, 2)(int64[:], int64[:], int64[:])")
def calculate_true_positive_rate(
    predictions: np.array, groups: np.array, labels: np.array
) -> (float, float):
    """
    Calculate the True Positive Rate (TPR) for two groups based on the predictions and labels.
    TPR = TP / (TP + FN)

    Parameters:
        predictions (np.array): An array of predictions.
        groups (np.array): An array of group identifiers.
        y (np.array): An array of true labels.

    Returns:
        (float, float): The TPR for the underpreviledged group and the TPR for the previledged group.
    """
    true_positive_1 = true_positive_2 = false_negative_1 = false_negative_2 = 0

    assert predictions.shape[0] == groups.shape[0] == labels.shape[0]

    for idx in prange(len(predictions)):
        if predictions[idx] != POSITIVE_OUTCOME:
            if predictions[idx] != NEGATIVE_OUTCOME:
                raise ValueError("Prediction must be 0 or 1.")

        if labels[idx] != POSITIVE_OUTCOME:
            if labels[idx] != NEGATIVE_OUTCOME:
                raise ValueError("Label must be 0 or 1.")

        if groups[idx] == UNPREVILEDGED:
            if labels[idx] == POSITIVE_OUTCOME:
                if predictions[idx] == POSITIVE_OUTCOME:
                    true_positive_1 += 1
                else:
                    false_negative_1 += 1
        elif groups[idx] == PREVILEDGED:
            if labels[idx] == POSITIVE_OUTCOME:
                if predictions[idx] == POSITIVE_OUTCOME:
                    true_positive_2 += 1
                else:
                    false_negative_2 += 1
        else:
            raise ValueError("Group identifier must be 0 or 1.")

    div_1 = true_positive_1 + false_negative_1
    div_2 = true_positive_2 + false_negative_2

    if div_1 == 0:
        div_1 = 1
    if div_2 == 0:
        div_2 = 1

    true_positive_rate_1 = true_positive_1 / div_1
    true_positive_rate_2 = true_positive_2 / div_2

    return (true_positive_rate_1, true_positive_rate_2)


@njit("float64[::1](float64[:], int64[:])", parallel=True)
def run_all_DSP_thresholds(predictions: np.array, list_protected: np.array) -> np.array:
    """
    Runs all DSP thresholds on the given predictions and list of protected values.

    Args:
        predictions (np.array): An array of predictions.
        list_protected (np.array): An array of protected values.

    Returns:
        np.array: An array containing demographic statistical parity values for each threshold.
    """
    return_array = np.empty(
        101,
    )

    thresholds = np.arange(0.0, 1.01, 0.01)
    for i in prange(len(thresholds)):
        return_array[i] = calculcate_demografic_statistical_parity(
            np.where(predictions >= thresholds[i], 1, 0), list_protected
        )

    return return_array


@njit("float64[:, ::1](float64[:], int64[:], int64[:])", parallel=True)
def run_all_TRP_thresholds(
    predictions: np.array, list_protected: np.array, labels: np.array
) -> np.array:
    """
    A function that runs all threshold values for a given set of predictions, list of protected attributes, and labels.

    Parameters:
        predictions (np.array): An array of predicted values.
        list_protected (np.array): An array of protected attributes.
        labels (np.array): An array of true labels.

    Returns:
        np.array: An array of true positive rates for each threshold value.
    """
    return_array = np.empty(
        (
            101,
            2,
        )
    )

    thresholds = np.arange(0.0, 1.01, 0.01)

    for i in prange(len(thresholds)):
        return_array[i] = calculate_true_positive_rate(
            np.where(predictions >= thresholds[i], 1, 0),
            np.where(list_protected == 1, 1, 0),
            np.where(labels == 1, 1, 0),
        )

    return return_array
