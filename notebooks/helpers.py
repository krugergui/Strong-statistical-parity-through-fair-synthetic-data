import os
from shutil import rmtree

from autogluon.tabular import TabularPredictor


def train_fast_predictor(data, path, label, hyperparameters=None):
    """
    Trains a tabular predictor using the given data and saves it to the specified path.

    Parameters:
        data (pandas.DataFrame): The training data.
        path (str): The path to save the trained predictor.
        hyperparameters (dict): The hyperparameters to use for training.

    Returns:
        TabularPredictor: The trained predictor.
    """
    return TabularPredictor(
        label=label,
        eval_metric="roc_auc",
        path=path,
        problem_type="binary",
    ).fit(
        train_data=data,
        num_bag_folds=0,
        num_bag_sets=0,
        num_stack_levels=0,
        hyperparameters=hyperparameters,
        presets="medium_quality",
        verbosity=1,
    )


def get_or_train_model(
    path, data=None, label=None, hyperparameters=None, run_from_scratch=False
):
    """
    Get the model if it exists, otherwise train it

    Args:
        data (pd.DataFrame): The data to train the model on
        path (str): The path to the model

    Returns:
        TabularPredictor: The trained model
    """
    if run_from_scratch or not os.path.isdir(path):
        try:
            rmtree(path)
        except FileNotFoundError:
            pass

        return train_fast_predictor(data, path, label, hyperparameters)
    else:
        return TabularPredictor.load(path, verbosity=1)
