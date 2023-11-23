import numpy as np
from numba import njit, prange

PREVILEDGED = 1
UNPREVILEDGED = 0
POSITIVE_OUTCOME = 1
NEGATIVE_OUTCOME = 0


@njit("Tuple((int64, int64, int64, int64))(int64[:], int64[:])", parallel=True, cache=True)
def get_groups_stats(predictions: np.array, groups: np.array):
    total_previledged, total_unpreviledged = 0, 0
    positive_unpreviledged, positive_previledged = 0, 0

    for idx in prange(len(predictions)):
        if groups[idx] == UNPREVILEDGED:
            total_unpreviledged += 1
            if predictions[idx] == POSITIVE_OUTCOME:
                positive_unpreviledged += 1
        elif groups[idx] == PREVILEDGED:
            total_previledged += 1
            if predictions[idx] == POSITIVE_OUTCOME:
                positive_previledged += 1

    return total_previledged, total_unpreviledged, positive_unpreviledged, positive_previledged


@njit("UniTuple(int64, 8)(int64[:], int64[:], int64[:])", cache=True, parallel=True)
def calculate_confusion_matrix(y_true, y_pred, groups):
    """
    Calculate True Positives (TP), True Negatives (TN),
    False Positives (FP), and False Negatives (FN) for two groups.

    Parameters:
    - y_true: Numpy array with true labels (0 or 1).
    - y_pred: Numpy array with predicted labels (0 or 1).

    Returns:
    A tuple (TP underpreviledged, TN underpreviledged, FP underpreviledged, FN underpreviledged, TP previledged, TN previledged, FP previledged, FN previledged).
    """
    TP1, TN1, FP1, FN1 = 0, 0, 0, 0
    TP2, TN2, FP2, FN2 = 0, 0, 0, 0

    for idx in prange(len(y_true)):
        if groups[idx] == UNPREVILEDGED:
            if y_true[idx] == POSITIVE_OUTCOME:
                if y_pred[idx] == POSITIVE_OUTCOME:
                    TP1 += 1
                else:
                    FN1 += 1
            elif y_pred[idx] == NEGATIVE_OUTCOME:
                TN1 += 1
            else:
                FP1 += 1
        else:
            if y_true[idx] == POSITIVE_OUTCOME:
                if y_pred[idx] == POSITIVE_OUTCOME:
                    TP2 += 1
                else:
                    FN2 += 1
            elif y_pred[idx] == NEGATIVE_OUTCOME:
                TN2 += 1
            else:
                FP2 += 1

    return TP1, TN1, FP1, FN1, TP2, TN2, FP2, FN2


@njit("UniTuple(float64, 2)(int64[:], int64[:], int64[:])", cache=True)
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

    assert predictions.shape[0] == groups.shape[0] == labels.shape[0]

    TP1, _, _, FN1, TP2, _, _, FN2 = calculate_confusion_matrix(labels, predictions, groups)

    div_1 = TP1 + FN1
    div_2 = TP2 + FN2

    if div_1 == 0:
        div_1 = 1

    if div_2 == 0:
        div_2 = 1

    true_positive_rate_1 = TP1 / div_1
    true_positive_rate_2 = TP2 / div_2

    return (true_positive_rate_1, true_positive_rate_2)


@njit("UniTuple(float64, 2)(int64[:], int64[:], int64[:])", cache=True)
def calculate_false_positive_rate(
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

    assert predictions.shape[0] == groups.shape[0] == labels.shape[0]

    TP1, TN1, FP1, FN1, TP2, TN2, FP2, FN2 = calculate_confusion_matrix(labels, predictions, groups)

    div_1 = FP1 + TN1
    div_2 = FP2 + TN2

    if div_1 == 0:
        div_1 = 1

    if div_2 == 0:
        div_2 = 1

    true_positive_rate_1 = FP1 / div_1
    true_positive_rate_2 = FP2 / div_2

    return (true_positive_rate_1, true_positive_rate_2)


@njit("float64(int64[:], int64[:])", cache=True)
def calculate_demografic_statistical_parity_difference(predictions, groups) -> float:
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

    (
        total_previledged,
        total_unpreviledged,
        positive_unpreviledged,
        positive_previledged,
    ) = get_groups_stats(predictions, groups)

    p_previledged = positive_previledged / total_previledged
    p_unpreviledged = positive_unpreviledged / total_unpreviledged

    return np.abs(p_previledged - p_unpreviledged)


@njit("float64(int64[:], int64[:], int64[:])", cache=True)
def calculate_equal_oportunity_difference(
    y_pred: np.array, y_true: np.array, groups: np.array
) -> float:
    """
    Calculate the Equal Opportunity Difference (EOD) for two groups based on the predictions and labels.
    EOD = TPR(A) - TPR(B)

    Parameters:
        y_pred (np.array): An array of predictions.
        y_true (np.array): An array of true labels.
        groups (np.array): An array of group identifiers.

    Returns:
        float: The EOD for the underpreviledged group and the EOD for the previledged group.
    """

    assert y_pred.shape[0] == y_true.shape[0] == groups.shape[0]

    TPR1, TPR2 = calculate_true_positive_rate(y_pred, groups, y_true)

    return TPR1 - TPR2


@njit("float64(int64[:], int64[:], int64[:])", cache=True)
def calculate_average_odds_difference(
    y_pred: np.array, y_true: np.array, groups: np.array
) -> float:
    """
    Calculate the average Equal Opportunity Difference (EOD) for two groups based on the predictions and labels.
    EOD = (( FPR(A) - FPR(B) ) + ( TPR(A) - TPR(B) )) / 2

    Parameters:
        y_pred (np.array): An array of predictions.
        y_true (np.array): An array of true labels.
        groups (np.array): An array of group identifiers.

    Returns:
        float: The EOD for the underpreviledged group and the EOD for the previledged group.
    """

    assert y_pred.shape[0] == y_true.shape[0] == groups.shape[0]

    FPR1, FPR2 = calculate_false_positive_rate(y_pred, groups, y_true)
    TPR1, TPR2 = calculate_true_positive_rate(y_pred, groups, y_true)

    return ((FPR1 - FPR2) + (TPR1 - TPR2)) / 2


@njit("float64(int64[:], int64[:])", cache=True)
def calculate_disparate_impact(predictions: np.array, groups: np.array) -> float:
    """
    Calculate the Disparate Impact of a predictive model.
    P ( Y = 1 | S=s1) / P ( Y = 1 | S=s2)

    Args:
        predictions (np.array): The predictions of the model. 1 for positive and 0 for negative.
        list_protected (np.array): The groups of the individuals. 1 for previledged and 0 for unpreviledged.

    Returns:
        float: The Disparate Impact score.
    """

    assert predictions.shape[0] == groups.shape[0]

    (
        total_previledged,
        total_unpreviledged,
        positive_unpreviledged,
        positive_previledged,
    ) = get_groups_stats(predictions, groups)

    if total_previledged == 0:
        total_previledged = 1

    if total_unpreviledged == 0:
        total_unpreviledged = 1

    p_previledged = positive_previledged / total_previledged
    p_unpreviledged = positive_unpreviledged / total_unpreviledged

    if p_unpreviledged == 0:
        return 0

    return p_previledged / p_unpreviledged


@njit("float64[::1](float64[:], int64[:])", parallel=True, cache=True)
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
        return_array[i] = calculate_demografic_statistical_parity_difference(
            np.where(predictions >= thresholds[i], 1, 0), list_protected
        )

    return return_array


@njit("float64[::1](float64[:], int64[:], int64[:])", parallel=True, cache=True)
def run_all_EOD_thresholds(
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
    return_array = np.empty((101,))

    thresholds = np.arange(0.0, 1.01, 0.01)

    for i in prange(len(thresholds)):
        return_array[i] = calculate_equal_oportunity_difference(
            np.where(predictions >= thresholds[i], 1, 0),
            list_protected,
            labels,
        )

    return return_array


@njit("float64[::1](float64[:], int64[:], int64[:])", parallel=True, cache=True)
def run_all_AOD_thresholds(
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
    return_array = np.empty((101,))

    thresholds = np.arange(0.0, 1.01, 0.01)

    for i in prange(len(thresholds)):
        return_array[i] = calculate_average_odds_difference(
            np.where(predictions >= thresholds[i], 1, 0),
            list_protected,
            labels,
        )

    return return_array


@njit("float64[::1](float64[:], int64[:])", parallel=True, cache=True)
def run_all_DID_thresholds(predictions: np.array, list_protected: np.array) -> np.array:
    """
    A function that runs all threshold values for a given set of predictions, list of protected attributes, and labels.

    Parameters:
        predictions (np.array): An array of predicted values.
        list_protected (np.array): An array of protected attributes.
        labels (np.array): An array of true labels.

    Returns:
        np.array: An array of true positive rates for each threshold value.
    """
    return_array = np.empty((101,))

    thresholds = np.arange(0.0, 1.01, 0.01)

    for i in prange(len(thresholds)):
        return_array[i] = calculate_disparate_impact(
            np.where(predictions >= thresholds[i], 1, 0),
            list_protected,
        )

    return return_array
