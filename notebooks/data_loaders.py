import pandas as pd


class AdultDataset:
    def load_data(self):
        test_data = pd.read_csv(
            "../Datasets/Adult-UCI/adult.test",
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
            "../Datasets/Adult-UCI/adult.data",
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
        df[obj_columns] = df[obj_columns].apply(
            lambda x: x.str.replace(".", "", regex=False)
        )

        # drop the fnlwgt and education-num columns to keep in line with the paper
        return df.drop(columns=["fnlwgt", "education-num"])

    def get_train_test_data(self):
        df = self.df
        test_size = 0.2  # 20% of the data will be used for testing
        test_df = df.sample(frac=test_size)
        train_df = df.drop(test_df.index)  # remove the test data from the original data

        return train_df, test_df

    def __init__(self):
        self.df = self.load_data()
