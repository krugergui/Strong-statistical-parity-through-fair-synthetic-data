import unittest

import numpy as np
import pandas as pd
from numpy import testing

from notebooks.fairness_metrics import (
    calculate_confusion_matrix,
    calculate_demografic_statistical_parity_difference,
    calculate_disparate_impact,
    calculate_equal_oportunity_difference,
    calculate_false_positive_rate,
    get_groups_stats,
    run_all_AOD_thresholds,
)

POSITIVE_OUTCOME, PREVILEDGED = 1, 1
NEGATIVE_OUTCOME, UNPREVILEDGED = 0, 0


class TestDSP(unittest.TestCase):
    def test_same_predictions(self):
        """
        Test the same predictions function.

        This function tests the same predictions function. It creates two numpy arrays,
        `predictions` and `groups`, with some predefined values. It then calls the `DSP`
        function with these arrays as arguments and asserts that the result is equal to 0.0
        using the `self.assertAlmostEqual()` method.
        """
        predictions = np.array([1, 1, 0, 0])
        groups = np.array([1, 0, 1, 0])
        self.assertAlmostEqual(
            calculate_demografic_statistical_parity_difference(predictions, groups), 0.0
        )

    def test_different_predictions(self):
        """
        Test the different predictions function.

        This function tests the different predictions function. It creates two numpy arrays,
        `predictions` and `groups`, with some predefined values. It then calls the `DSP`
        function with these arrays as arguments and asserts that the result is equal to 1.0
        using the `self.assertAlmostEqual()` method.
        """
        predictions = np.array([1, 1, 1, 0, 0])
        groups = np.array([1, 1, 1, 0, 0])
        self.assertAlmostEqual(
            calculate_demografic_statistical_parity_difference(predictions, groups), 1.0
        )

    def test_mixed_predictions(self):
        """
        Test the mixed predictions function.

        This function tests the mixed predictions function. It creates two numpy arrays,
        `predictions` and `groups`, with some predefined values. It then calls the `DSP`
        function with these arrays as arguments and asserts that the result is equal to 1/6
        using the `self.assertAlmostEqual()` method.
        """
        predictions = np.array([1, 0, 1, 0, 1])
        groups = np.array([1, 1, 0, 0, 1])
        self.assertAlmostEqual(
            calculate_demografic_statistical_parity_difference(predictions, groups), 1 / 6
        )

    def test_no_previledged_group(self):
        """
        Test the no previledged group function.

        This function tests the no previledged group function. It creates two numpy arrays,
        `predictions` and `groups`, with some predefined values. It then calls the `DSP`
        function with these arrays as arguments and asserts that a `ZeroDivisionError`
        exception is raised.
        """
        predictions = np.array([1, 0, 1, 0, 1])
        groups = np.array([0, 0, 0, 0, 0])
        try:
            self.assertAlmostEqual(
                calculate_demografic_statistical_parity_difference(predictions, groups), -1.0
            )
            assert False
        except ZeroDivisionError:
            assert True

    def test_no_unpreviledged_group(self):
        """
        Test the no unpreviledged group function.

        This function tests the no unpreviledged group function. It creates two numpy arrays,
        `predictions` and `groups`, with some predefined values. It then calls the `DSP`
        function with these arrays as arguments and asserts that a `ZeroDivisionError`
        exception is raised.
        """
        predictions = np.array([1, 0, 1, 0, 1])
        groups = np.array([1, 1, 1, 1, 1])
        try:
            self.assertAlmostEqual(
                calculate_demografic_statistical_parity_difference(predictions, groups), -1.0
            )
            assert False
        except ZeroDivisionError:
            assert True

    def test_empty_predictions(self):
        """
        Test the empty predictions function.

        This function tests the empty predictions function. It creates two empty numpy arrays,
        `predictions` and `groups`. It then calls the `DSP` function with these arrays as
        arguments and asserts that a `ZeroDivisionError` exception is raised.
        """
        predictions = np.array([], dtype="int64")
        groups = np.array([], dtype="int64")
        try:
            self.assertAlmostEqual(
                calculate_demografic_statistical_parity_difference(predictions, groups), -1.0
            )
            assert False
        except ZeroDivisionError:
            assert True

    def test_assert_data(self):
        """
        Test the assert data function.

        This function tests the assert data function. It loads the data using the
        load_data() function and asserts that the length of the data is equal to 48842.
        It then calculates the DSP (Data Science Power) using the np.where() function
        to filter the data based on the "income" and "sex" columns. The calculated DSP
        is compared to a known value of 0.1945157 using the self.assertAlmostEqual()
        function. This value is based on a reference table provided in the research paper
        by FERNANDO ET AL. [1].

        References:
        [1] FERNANDO ET AL. Table 1 - https://onlinelibrary.wiley.com/doi/pdf/10.1002/int.22415
        """
        data = load_data()
        self.assertEqual(len(data), 48842)
        self.assertAlmostEqual(
            calculate_demografic_statistical_parity_difference(
                np.where(data["income"] == ">50K", 1, 0),
                np.where(data["sex"] == "Female", 1, 0),
            ),
            0.1945157,  # According to FERNANDOET AL. Table 1 - https://onlinelibrary.wiley.com/doi/pdf/10.1002/int.22415
        )


def load_data():
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    test_data = pd.read_csv("Datasets/Adult-UCI/adult.test", header=None, skiprows=1, names=columns)
    train_data = pd.read_csv("Datasets/Adult-UCI/adult.data", header=None, names=columns)

    df = pd.concat([train_data, test_data], ignore_index=True)

    df.columns = df.columns.str.strip().str.replace(".", "", regex=False)

    obj_columns = df.select_dtypes(["object"]).columns
    df[obj_columns] = df[obj_columns].apply(
        lambda x: x.str.strip().str.replace(".", "", regex=False)
    )

    return df


class TestCalculateConfusionMatrix(unittest.TestCase):
    def test_case1(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        groups = np.array([0, 1, 0, 1])
        expected_result = (0, 1, 1, 0, 1, 0, 0, 1)
        self.assertEqual(calculate_confusion_matrix(y_true, y_pred, groups), expected_result)

    def test_case2(self):
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 0])
        groups = np.array([0, 1, 0, 1, 0])
        expected_result = (1, 1, 0, 1, 0, 1, 0, 1)
        self.assertEqual(calculate_confusion_matrix(y_true, y_pred, groups), expected_result)

    def test_case3(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0])
        groups = np.array([1, 1, 0, 0, 1])
        expected_result = (1, 0, 0, 1, 0, 0, 2, 1)
        self.assertEqual(calculate_confusion_matrix(y_true, y_pred, groups), expected_result)


def test_calculate_equal_oportunity_difference():
    y_pred = np.array([1, 0, 1, 0, 1])
    y_true = np.array([1, 1, 0, 0, 1])
    groups = np.array([0, 0, 1, 1, 1])

    result = calculate_equal_oportunity_difference(y_pred, y_true, groups)

    assert np.isclose(result, -0.5)

    y_pred = np.array([0, 1, 1, 0, 0])
    y_true = np.array([1, 0, 1, 1, 0])
    groups = np.array([0, 0, 1, 1, 1])

    result = calculate_equal_oportunity_difference(y_pred, y_true, groups)

    assert np.isclose(result, -0.5)

    y_pred = np.array([0, 0, 0, 0, 0])
    y_true = np.array([1, 1, 1, 1, 1])
    groups = np.array([0, 0, 1, 1, 1])

    result = calculate_equal_oportunity_difference(y_pred, y_true, groups)

    assert np.isclose(result, 0.0)


def test_calculate_false_positive_rate():
    predictions = np.array([1, 0, 0, 1])
    groups = np.array([0, 1, 1, 1])
    labels = np.array([0, 1, 1, 0])

    tpr_1, tpr_2 = calculate_false_positive_rate(predictions, groups, labels)
    testing.assert_almost_equal(tpr_1, 1.0)
    testing.assert_almost_equal(tpr_2, 1.0)

    predictions = np.array([1, 0, 1, 0])
    groups = np.array([0, 0, 1, 1])
    labels = np.array([1, 1, 1, 0])

    tpr_1, tpr_2 = calculate_false_positive_rate(predictions, groups, labels)
    testing.assert_almost_equal(tpr_1, 0)
    testing.assert_almost_equal(tpr_2, 0)


def test_run_all_AOD_thresholds():
    # Test case 1
    predictions = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    list_protected = np.array([0, 1, 0, 1, 0])
    labels = np.array([0, 1, 1, 0, 1])
    result = run_all_AOD_thresholds(predictions, list_protected, labels)
    testing.assert_equal(np.sum(result), -10.0)
    testing.assert_almost_equal(np.std(result), 0.2538788)

    # Test case 2
    predictions = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    list_protected = np.array([1, 0, 1, 0, 1])
    labels = np.array([1, 0, 0, 1, 0])
    result = run_all_AOD_thresholds(predictions, list_protected, labels)
    testing.assert_equal(np.sum(result), 10.5)
    testing.assert_almost_equal(np.std(result), 0.2518921)

    # Test case 3
    predictions = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    list_protected = np.array([0, 0, 0, 0, 0])
    labels = np.array([0, 0, 0, 0, 0])
    result = run_all_AOD_thresholds(predictions, list_protected, labels)
    testing.assert_almost_equal(np.sum(result), 20.5)
    testing.assert_almost_equal(np.std(result), 0.143820188)

    # Test case 4
    predictions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    list_protected = np.array([1, 1, 1, 1, 1])
    labels = np.array([1, 1, 1, 1, 1])
    result = run_all_AOD_thresholds(predictions, list_protected, labels)
    testing.assert_equal(np.sum(result), -50.5)
    testing.assert_almost_equal(np.std(result), 0)


class TestGetGroupsStats(unittest.TestCase):
    def test_empty_arrays(self):
        predictions = np.array([], dtype=int)
        groups = np.array([], dtype=int)
        result = get_groups_stats(predictions, groups)
        self.assertEqual(result, (0, 0, 0, 0))

    def test_positive_outcomes_unprivileged(self):
        predictions = np.array([POSITIVE_OUTCOME] * 10)
        groups = np.array([UNPREVILEDGED] * 10)
        result = get_groups_stats(predictions, groups)
        self.assertEqual(result, (0, 10, 10, 0))

    def test_positive_outcomes_privileged(self):
        predictions = np.array([POSITIVE_OUTCOME] * 10)
        groups = np.array([PREVILEDGED] * 10)
        result = get_groups_stats(predictions, groups)
        self.assertEqual(result, (10, 0, 0, 10))

    def test_mixed_outcomes(self):
        predictions = np.array(
            [POSITIVE_OUTCOME, NEGATIVE_OUTCOME, POSITIVE_OUTCOME, NEGATIVE_OUTCOME]
        )
        groups = np.array([UNPREVILEDGED, UNPREVILEDGED, PREVILEDGED, PREVILEDGED])
        result = get_groups_stats(predictions, groups)
        self.assertEqual(result, (2, 2, 1, 1))


import numpy as np


def test_calculate_disparate_impact():
    # Test case 1: No positive predictions
    predictions = np.array([0, 1, 1, 0, 0])
    groups = np.array([1, 1, 0, 0, 0])
    assert calculate_disparate_impact(predictions, groups) == 1.5

    # Test case 2: No unprivileged individuals
    predictions = np.array([1, 1, 1, 1, 1])
    groups = np.array([1, 1, 1, 1, 1])
    assert calculate_disparate_impact(predictions, groups) == 0

    # Test case 3: Half positive predictions for privileged and unprivileged individuals
    predictions = np.array([0, 0, 1, 1, 0, 1])
    groups = np.array([1, 1, 0, 0, 1, 0])
    assert calculate_disparate_impact(predictions, groups) == 0.0

    # Test case 4: All positive predictions for privileged individuals
    predictions = np.array([1, 1, 1, 1, 1])
    groups = np.array([1, 1, 0, 0, 0])
    testing.assert_equal(calculate_disparate_impact(predictions, groups), 1.0)

    # Test case 5: All positive predictions for unprivileged individuals
    predictions = np.array([1, 1, 1, 1, 1])
    groups = np.array([1, 1, 1, 1, 0])
    assert calculate_disparate_impact(predictions, groups) == 1.0


if __name__ == "__main__":
    unittest.main()
