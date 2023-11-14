import pandas as pd
from notebooks.fairness_metrics import DSP
import unittest


class TestDSP(unittest.TestCase):
    def test_same_predictions(self):
        predictions = [1, 1, 0, 0]
        groups = [1, 0, 1, 0]
        self.assertAlmostEqual(DSP(predictions, groups), 0.0)

    def test_different_predictions(self):
        predictions = [1, 1, 1, 0, 0]
        groups = [1, 1, 1, 0, 0]
        self.assertAlmostEqual(DSP(predictions, groups), 1.0)

    def test_mixed_predictions(self):
        predictions = [1, 0, 1, 0, 1]
        groups = [1, 1, 0, 0, 1]
        self.assertAlmostEqual(DSP(predictions, groups), 1 / 6)

    def test_no_previledged_group(self):
        predictions = [1, 0, 1, 0, 1]
        groups = [0, 0, 0, 0, 0]
        self.assertAlmostEqual(DSP(predictions, groups), -1.0)

    def test_no_unpreviledged_group(self):
        predictions = [1, 0, 1, 0, 1]
        groups = [1, 1, 1, 1, 1]
        self.assertAlmostEqual(DSP(predictions, groups), -1.0)

    def test_empty_predictions(self):
        predictions = []
        groups = []
        self.assertAlmostEqual(DSP(predictions, groups), -1.0)

    def test_assert_data(self):
        data = load_data()
        self.assertEqual(len(data), 48842)
        self.assertAlmostEqual(
            DSP(data["income"], data["sex"], ">50K", "Male"),
            0.1945157,  # According to FERNANDOET AL. Table 1 - https://onlinelibrary.wiley.com/doi/pdf/10.1002/int.22415
        )


def load_data():
    test_data = pd.read_csv(
        "Datasets/Adult-UCI/adult.test",
        header=None,
        skiprows=1,
        names=[
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
        ],
    )

    train_data = pd.read_csv(
        "Datasets/Adult-UCI/adult.data",
        header=None,
        names=[
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
        ],
    )

    df = pd.concat([train_data, test_data]).reset_index(drop=True)

    df.columns = df.columns.str.strip()

    obj_columns = df.select_dtypes(["object"]).columns
    df[obj_columns] = df[obj_columns].apply(lambda x: x.str.strip())
    df[obj_columns] = df[obj_columns].apply(lambda x: x.str.replace(".", ""))

    return df


if __name__ == "__main__":
    unittest.main()
