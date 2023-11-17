import unittest

import numpy as np
import pandas as pd

from notebooks.fairness_metrics import calculcate_demografic_statistical_parity


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
            calculcate_demografic_statistical_parity(predictions, groups), 0.0
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
            calculcate_demografic_statistical_parity(predictions, groups), 1.0
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
            calculcate_demografic_statistical_parity(predictions, groups), 1 / 6
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
                calculcate_demografic_statistical_parity(predictions, groups), -1.0
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
                calculcate_demografic_statistical_parity(predictions, groups), -1.0
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
                calculcate_demografic_statistical_parity(predictions, groups), -1.0
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
            calculcate_demografic_statistical_parity(
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

    test_data = pd.read_csv(
        "Datasets/Adult-UCI/adult.test", header=None, skiprows=1, names=columns
    )
    train_data = pd.read_csv(
        "Datasets/Adult-UCI/adult.data", header=None, names=columns
    )

    df = pd.concat([train_data, test_data], ignore_index=True)

    df.columns = df.columns.str.strip().str.replace(".", "")

    obj_columns = df.select_dtypes(["object"]).columns
    df[obj_columns] = df[obj_columns].apply(
        lambda x: x.str.strip().str.replace(".", "")
    )

    return df


if __name__ == "__main__":
    unittest.main()
