"""A module for the class for loading and preprocessing the Adult-UCI dataset.
"""

import pandas as pd


class AdultDataset:
    """
    A class for loading and preprocessing the Adult-UCI dataset.

    Methods:
    - load_data: Load data from the Adult-UCI dataset.
    - get_train_test_data: Generate train and test data for the adult dataframe.
    """

    def load_data(self):
        """
        Load data from the Adult-UCI dataset.

        This function reads the data from the 'adult.test' and 'adult.data' files in the
        '../Datasets/Adult-UCI/' directory.  It selects the relevant columns and
        concatenates the train and test data into a single DataFrame.
        The column names are stripped of leading and trailing spaces.
        The object columns are stripped of leading and trailing spaces, and any '.'
        characters are removed.
        The 'fnlwgt' and 'education-num' columns are dropped from the DataFrame.

        Returns:
        - df (pandas.DataFrame): The loaded and preprocessed data.
        """
        relevant_columns = [
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
            "../Datasets/Adult-UCI/adult.test",
            header=None,
            skiprows=1,
            names=relevant_columns,
        )

        train_data = pd.read_csv(
            "../Datasets/Adult-UCI/adult.data",
            header=None,
            names=relevant_columns,
        )

        df = pd.concat([train_data, test_data]).reset_index(drop=True)

        df.columns = df.columns.str.strip()

        obj_columns = df.select_dtypes(["object"]).columns
        df[obj_columns] = df[obj_columns].apply(lambda x: x.str.strip())
        df[obj_columns] = df[obj_columns].apply(lambda x: x.str.replace(".", "", regex=False))

        # drop the fnlwgt and education-num columns to keep in line with the paper
        return df.drop(columns=["fnlwgt", "education-num"])

    def get_train_test_data(self, random_state=42):
        """
        Generate train and test data for the adult dataframe.

        Parameters:
            random_state (int): Random seed for reproducible results. Defaults to 42.

        Returns:
            tuple: A tuple containing train and test dataframes.
        """
        df = self.df
        test_size = 0.2  # 20% of the data will be used for testing
        test_df = df.sample(frac=test_size, random_state=random_state)
        train_df = df.drop(test_df.index)  # remove the test data from the original data

        return train_df, test_df

    def __init__(self):
        """
        Initializes a new instance of the class.

        Parameters:
            None

        Returns:
            None
        """
        self.df = self.load_data()
